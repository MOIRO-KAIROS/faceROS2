# Copyright (C) 2023  Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import cv2
import numpy as np
from typing import Tuple

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import message_filters
from cv_bridge import CvBridge
from ultralytics.utils.plotting import Annotator, colors
# import time

from sensor_msgs.msg import Image
from moiro_interfaces.msg import KeyPoint2D
from moiro_interfaces.msg import Detection, DetectionArray

class DebugNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("debug_node")
        self.cv_bridge = CvBridge()

        # params
        self.declare_parameter("image_reliability",
                               QoSReliabilityPolicy.BEST_EFFORT)
        
        self.get_logger().info("Debug node created")

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Configuring {self.get_name()}')

        self.image_qos_profile = QoSProfile(
            reliability=self.get_parameter(
                "image_reliability").get_parameter_value().integer_value,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        # pubs
        self._dbg_pub = self.create_publisher(Image, "dbg_image", 10)
    
        return TransitionCallbackReturn.SUCCESS
        
    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Activating {self.get_name()}')
        
        # subs
        image_sub = message_filters.Subscriber(
            self, Image, "image_raw", qos_profile=self.image_qos_profile)
        detections_sub = message_filters.Subscriber(
            self, DetectionArray, "detections", qos_profile=10)
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (image_sub, detections_sub), 100, 0.01)
        
        #self._synchronizer = message_filters.ApproximateTimeSynchronizer((image_sub, detections_sub), 10, 0.5)
        self._synchronizer.registerCallback(self.detections_cb)
        return TransitionCallbackReturn.SUCCESS
    
    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Deactivating {self.get_name()}')

        self.destroy_subscription(self.image_sub.sub)
        self.destroy_subscription(self.detections_sub.sub)

        del self._synchronizer

        return TransitionCallbackReturn.SUCCESS
    
    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Cleaning up {self.get_name()}')

        self.destroy_publisher(self._dbg_pub)

        return TransitionCallbackReturn.SUCCESS

    def draw_box(self, cv_image: np.array, detection: Detection) -> np.array:
        # draw person box
        cv2.rectangle(cv_image, detection.bboxyolo.leftup, detection.bboxyolo.rightbottom, (135, 204, 255), 2)

        # draw face box
        if detection.facebox.isdetect: # 얼굴 좌표가 없다 = 얼굴 이름도 없다 하지만 바운딩박스의 이름은 array 있으면 찾을 수 있다.
            cv2.rectangle(cv_image, detection.facebox.bbox.leftup, detection.facebox.bbox.rightbottom, (255,255,255), 2)

        # write face's name
        font = cv2.FONT_HERSHEY_COMPLEX
        pos = (detection.bboxyolo.leftup[0] + 25, detection.bboxyolo.rightbottom[1] - 25)
        
        label = "({}) {}".format(detection.id, detection.facebox.name)
        if detection.facebox.name == "unknown":
            cv2.putText(cv_image, label, pos, font,
                0.6, (255,255,255), 1, cv2.LINE_AA)
        elif detection.facebox.name == "no face":
            cv2.putText(cv_image, label, pos, font,
                0.6, (135, 204, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(cv_image, label, pos, font,
                0.6, (0,255,0), 1, cv2.LINE_AA)

        return cv_image

    def draw_mask(self, cv_image: np.array, detection: Detection) -> np.array:

        mask_msg = detection.mask
        mask_array = np.array([[int(ele.x), int(ele.y)]
                              for ele in mask_msg.data])

        if mask_msg.data:
            layer = cv_image.copy()
            layer = cv2.fillPoly(layer, pts=[mask_array], color=(135, 204, 255))
            cv2.addWeighted(cv_image, 0.4, layer, 0.6, 0, cv_image)
            cv_image = cv2.polylines(cv_image, [mask_array], isClosed=True,
                                     color=(135, 204, 255), thickness=2, lineType=cv2.LINE_AA)
        return cv_image

    def draw_keypoints(self, cv_image: np.array, detection: Detection) -> np.array:
        keypoints_msg = detection.keypoints

        ann = Annotator(cv_image)
        sh_point = [0,0]
        kp: KeyPoint2D
        cnt = 0
        for kp in keypoints_msg.data:
            #### Shoulder middle point!
            if str(kp.id) == '7' or str(kp.id) == '6':
                cnt+=1
                sh_point[0] += kp.point.x
                sh_point[1] += kp.point.y
        if cnt!=0:
            sh_point[0],sh_point[1] = sh_point[0]//cnt,sh_point[1]//cnt
        cv2.circle(cv_image, (int(sh_point[0]), int(sh_point[1])),
                       5, (255,255,255), -1, lineType=cv2.LINE_AA)

        return cv_image , sh_point


    def detections_cb(self, img_msg: Image, face_detection_msg: DetectionArray) -> None:
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg,"rgb8")
        detection: Detection
        for detection in face_detection_msg.detections:
            cv_image, sh_point = self.draw_keypoints(cv_image, detection)
            cv_image = self.draw_box(cv_image, detection)
            
        # publish dbg image
        self._dbg_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_image, encoding=img_msg.encoding))


def main():
    rclpy.init()
    node = DebugNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()