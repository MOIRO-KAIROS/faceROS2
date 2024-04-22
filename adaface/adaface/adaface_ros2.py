import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

import message_filters
from cv_bridge import CvBridge

from yolov8_msgs.msg import Detection
from yolov8_msgs.msg import DetectionArray
from sensor_msgs.msg._image import Image

import sys
sys.path.append("/home/lee52/ros2_ws/src/minhaROS/adaface/adaface/script")

from adaface.script.adaface import AdaFace

'''
This Node subscribes datas from ~yolo/tracking_node~, publish data to ~yolo/debug_node~
'''

class Adaface(Node):
  def __init__(self):
    super().__init__('adaface')
    self.get_logger().info('========================') 
    self.get_logger().info('Start face recognition!')
    self.get_logger().info('========================') 

    self._class_to_color = {}
    self.cv_bridge = CvBridge()
    
    # params
    self.declare_parameter("fr_weight", "ir_50")
    model = self.get_parameter("fr_weight").get_parameter_value().string_value

    self.declare_parameter("device", "cuda:0")
    self.device = self.get_parameter("device").get_parameter_value().string_value

    self.declare_parameter("option", 1)  
    option = self.get_parameter("option").get_parameter_value().integer_value
    
    self.declare_parameter("thresh", 0.2)
    self.thresh = self.get_parameter("thresh").get_parameter_value().double_value

    self.declare_parameter("max_obj", 6)        
    self.max_obj = self.get_parameter("max_obj").get_parameter_value().integer_value

    self.declare_parameter("dataset", "face_dataset/test")  
    self.dataset = self.get_parameter("dataset").get_parameter_value().string_value
    
    self.declare_parameter("video", "0")
    self.video = self.get_parameter("video").get_parameter_value().string_value
    
    self.declare_parameter("image_reliability",
                            QoSReliabilityPolicy.BEST_EFFORT)
    image_qos_profile = QoSProfile(
        reliability=self.get_parameter(
            "image_reliability").get_parameter_value().integer_value,
        history=QoSHistoryPolicy.KEEP_LAST,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=1
    )

    self.cv_bridge = CvBridge()
    self.adaface = AdaFace(
        model=model,
        option=option,
        dataset=self.dataset,
        video=self.video,
        max_obj=self.max_obj,
        thresh=self.thresh,
    )
    
    self.get_logger().info('Adaface load suceess...')
    #pubs
    self._adaface_pub = self.create_publisher(DetectionArray, 'adaface_msg',10)

    ## subs
    # 이미지와 message를 subscribe
    tracking_sub = message_filters.Subscriber(
        self, DetectionArray, "/yolo/tracking", qos_profile =10)
    image_sub = message_filters.Subscriber(
        self, Image, "/camera/camera/color/image_raw", qos_profile=image_qos_profile)

    
    # 이미지와 message를 동기화
    self._synchronizer = message_filters.ApproximateTimeSynchronizer(
        (image_sub, tracking_sub), 10, 0.5)
    
    # Adaface_main 부름
    self._synchronizer.registerCallback(self.adaface_main)

  
  def adaface_main(self, img_msg: Image, tracking_msg: DetectionArray) -> None:
        
        face_ids_for_frame = DetectionArray()
        face_ids_for_frame.header = img_msg.header

        # convert image for align
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg)

        face_id_msg: Detection
        for face_id_msg in tracking_msg.detections:
          
          # 객체 이미지 위치 잡고 그걸 inference로 보낸다
          x1 = int(face_id_msg.bbox.center.position.x - face_id_msg.bbox.size.x / 2)
          y1 = int(face_id_msg.bbox.center.position.y - face_id_msg.bbox.size.y / 2)
          x2 = int(face_id_msg.bbox.center.position.x + face_id_msg.bbox.size.x / 2)
          y2 = int(face_id_msg.bbox.center.position.y + face_id_msg.bbox.size.y / 2)
          
          face_box, face_names = self.adaface.inference(cv_image[y1:y2,x1:x2])

          if face_box:
          # Assume that one person box = one face
            # print(type(x1 + (face_box[0][2] + face_box[0][0])//2))
            face_id_msg.bbox.center.position.x = float(x1 + (face_box[0][2] + face_box[0][0])//2)
            face_id_msg.bbox.size.x = float(face_box[0][2]- face_box[0][0])
            face_id_msg.bbox.center.position.y = float(y1 + (face_box[0][3] + face_box[0][1])//2)
            face_id_msg.bbox.size.y = float(face_box[0][3]- face_box[0][1])
            face_id_msg.id = str(face_names[0])

            face_ids_for_frame.detections.append(face_id_msg)
            self.get_logger().info('===============================================================')
            self.get_logger().info('center | x : {}, y : {}, Person Name | {}'.format(face_id_msg.bbox.center.position.x ,face_id_msg.bbox.center.position.y ,face_id_msg.id)) # For Debugging
            self.get_logger().info('===============================================================')
        # publish face information (id,bbox)
        self.get_logger().info('Publish data')
        self._adaface_pub.publish(face_ids_for_frame)

def main(args=None): 
  rclpy.init(args=None)
  node = Adaface()
  rclpy.spin(node)
  node.destroy_node()
  rclpy.shutdown()

if __name__ == '__main__':
  main()