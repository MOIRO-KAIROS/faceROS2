import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

from ament_index_python.packages import get_package_share_directory
import message_filters
from cv_bridge import CvBridge

from moiro_interfaces.msg import Detection
from moiro_interfaces.msg import DetectionArray
from sensor_msgs.msg._image import Image

import numpy as np
import sys
import os


# get package share example : /home/minha/moiro_ws/install/adaface_ros/lib/adaface_ros/
package_path = os.path.abspath(get_package_share_directory('adaface_ros')).split('install')[0]
script_path = "src/moiro_vision/adaface_ros/adaface_ros/script"
sys.path.append(os.path.join(package_path, script_path))

from adaface_ros.script.adaface  import AdaFace

'''
This Node subscribes datas from ~yolo/tracking_node~, publish data to ~yolo/debug_node~
'''

class Adaface_ros(LifecycleNode):
    def __init__(self)-> None:
        super().__init__('adaface')

        self.cv_bridge = CvBridge()
        self._face_cache = {}

        # params
        self.declare_parameter("fr_weight", "ir_50")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("option", 1) 
        self.declare_parameter("thresh", 0.2) 
        self.declare_parameter("max_obj", 6)  
        self.declare_parameter("dataset", "face_dataset/test")
        self.declare_parameter("video", 0)
        self.declare_parameter("image_reliability",
                                QoSReliabilityPolicy.BEST_EFFORT)     
        self.get_logger().info("Face Recognition node created")
    

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Configuring {self.get_name()}')

        self.model = self.get_parameter("fr_weight").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.option = self.get_parameter("option").get_parameter_value().integer_value
        self.thresh = self.get_parameter("thresh").get_parameter_value().double_value   
        self.max_obj = self.get_parameter("max_obj").get_parameter_value().integer_value
        self.dataset = self.get_parameter("dataset").get_parameter_value().string_value
        self.video = self.get_parameter("video").get_parameter_value().string_value
        
        # pubs
        self._adaface_pub = self.create_publisher(DetectionArray, 'adaface_msg',10)

        return TransitionCallbackReturn.SUCCESS
    
    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Activating {self.get_name()}')

        self.image_qos_profile = QoSProfile(
                reliability=self.get_parameter(
                    "image_reliability").get_parameter_value().integer_value,
                history=QoSHistoryPolicy.KEEP_LAST,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1
            )
        
        self.adaface = AdaFace(
            model=self.model,
            option=self.option,
            dataset=self.dataset,
            video=self.video,
            max_obj=self.max_obj,
            thresh=self.thresh,
        )
        
        ## subs
        tracking_sub = message_filters.Subscriber(
            self, DetectionArray, "detections", qos_profile =10)
        image_sub = message_filters.Subscriber(
            self, Image, "image_raw", qos_profile=self.image_qos_profile)

        self._synchronizer = message_filters.TimeSynchronizer(
            (image_sub, tracking_sub), 100)
        self._synchronizer.registerCallback(self.adaface_main)

        return TransitionCallbackReturn.SUCCESS
  
    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Deactivating {self.get_name()}')

        self.destroy_subscription(self.image_sub.sub)
        self.destroy_subscription(self.tracking_sub.sub)

        del self._synchronizer
        self._synchronizer = None

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Cleaning up {self.get_name()}')

        # self.destroy_publisher(self._adaface_pub)
        del self.image_qos_profile

        return TransitionCallbackReturn.SUCCESS
  
    def adaface_main(self, img_msg: Image, detections_msg: DetectionArray) -> None:

        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg)

        detection : Detection
        
        keys_to_keep = []
        for n, detection in enumerate(detections_msg.detections):
            # 객체 이미지 위치 잡고 그걸 inference로 보낸다
            x1 = int(np.clip((detection.bbox.center.position.x - detection.bbox.size.x / 2), 0, img_msg.width - 1)) # img_msg.width = 640 # ? 설정 시: 640 - 1
            y1 = int(np.clip((detection.bbox.center.position.y - detection.bbox.size.y / 2), 0, img_msg.height - 1)) # img_msg.height = 480
            x2 = int(np.clip((detection.bbox.center.position.x + detection.bbox.size.x / 2), 0, img_msg.width - 1))
            y2 = int(np.clip((detection.bbox.center.position.y + detection.bbox.size.y / 2), 0, img_msg.height - 1))

            detection.bboxyolo.leftup = [x1, y1]
            detection.bboxyolo.rightbottom = [x2, y2]
            face_box, face_info = self.adaface.inference(cv_image[y1:y2,x1:x2])

            if face_box: # Assume that one person box = one face
                detection.facebox.bbox.leftup = [x1 + face_box[0][0] , y1 + face_box[0][1]]
                detection.facebox.bbox.rightbottom = [x1 + face_box[0][2], y1 + face_box[0][3]]
                
                detection.facebox.name = face_info[0]
                detection.facebox.score = face_info[1]
                detection.facebox.isdetect = True            
            else:
                detection.facebox.isdetect = False
                detection.facebox.name = "no face"

            # dictionary 관련 + name update하기
            keys_to_keep.append(detection.id)
                
            if detection.facebox.name not in ["unknown", "no face"]:
                if detection.id not in self._face_cache.keys() or self._face_cache[detection.id][0] in ["unknown", "no face"]:
                    self._face_cache[detection.id] = [detection.facebox.name, 0]
                else: # penalty 증가 (minha != yeonju)
                    if self._face_cache[detection.id][0] != detection.facebox.name:
                        self._face_cache[detection.id][1] += 1
                    else: # 연속이 아니면 0으로 초기화
                        self._face_cache[detection.id][1] = 0
                    if self._face_cache[detection.id][1] == 3:
                        self._face_cache[detection.id] = [detection.facebox.name, 0]
            else:
                if detection.id not in self._face_cache.keys():
                    self._face_cache[detection.id] = [detection.facebox.name, 0]
            # dictionary 값을 실제 facebox.name을 반영
            detection.facebox.name = self._face_cache[detection.id][0]
            detections_msg.detections[n] = detection
        
        self._adaface_pub.publish(detections_msg)

        # 키 삭제
        keys_to_remove = [key for key in self._face_cache if key not in keys_to_keep]
        for key in keys_to_remove:
            del self._face_cache[key]


def main(args=None): 
    rclpy.init()
    node = Adaface_ros()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()