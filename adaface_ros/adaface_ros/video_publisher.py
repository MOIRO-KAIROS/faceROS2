import rclpy
from rclpy.node import Node
import cv2
import time, os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ament_index_python.packages import get_package_share_directory

class ImagePublisher(Node):
    def __init__(self):
        super().__init__("image_publisher")
        package_path = os.path.abspath(os.path.join(get_package_share_directory('adaface_ros'), "../../../../"))
        file_path = os.path.join(package_path,'src/moiro_vision/adaface_ros/adaface_ros/script/video/iAM.mp4')
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(file_path)
        self.pub = self.create_publisher(Image, "video_topic", 10)

    def run(self):
        while(self.cap.isOpened()):
            ret, frame = self.cap.read() 
            if ret:
                self.pub.publish(self.bridge.cv2_to_imgmsg(frame,"rgb8"))
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(1/60) 

        self.cap.release()

def main(args=None):
    rclpy.init(args=None)
    ip = ImagePublisher()
    print("Publishing...")
    ip.run()

    ip.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()