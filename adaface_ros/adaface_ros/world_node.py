import rclpy
from rclpy.node import Node
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from moiro_interfaces.msg import DetectionArray, Detection, KeyPoint2DArray
from moiro_interfaces.srv import Person, TargetPose
from geometry_msgs.msg import PoseStamped, TransformStamped, Pose
import transforms3d.quaternions as txq
import numpy as np

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import tf2_ros
from rclpy.executors import MultiThreadedExecutor

class WorldNode(Node):

    def __init__(self):
        super().__init__('world_node')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.cv_bridge = CvBridge()

        self.status = False
        
        self.person_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Parameters
        self.declare_parameter('depth_image_reliability', 1)  # QoSReliabilityPolicy.BEST_EFFORT, Default to BEST_EFFORT
        depth_image_qos_profile = QoSProfile(
            reliability=self.get_parameter(
                'depth_image_reliability').get_parameter_value().integer_value, 
            depth=1
        )
        srv_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10 
        )

        self.declare_parameter("person_name", 'Unintialized')
        self.person_name = self.get_parameter("person_name").get_parameter_value().string_value

        self.get_logger().info('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-') 
        self.get_logger().info(f'The Person Who You Want To Detect Is {self.person_name} !!!!')
        self.get_logger().info('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')

        # Subscribers
        self.depth_sub = message_filters.Subscriber(self, Image, 'depth_image', qos_profile=depth_image_qos_profile)
        self.detections_sub = message_filters.Subscriber(
            self, DetectionArray, "detections")
        
        # ApproximateTimeSynchronizer
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.detections_sub], queue_size=100, slop=0.1)
        self._synchronizer.registerCallback(self.person_tf)

        # services
        self._srv = self.create_service(Person, 'person_name', self.person_setting)
        self.target_server = self.create_service(TargetPose,'target_pose', self.target_setting, qos_profile=srv_qos_profile)

    def target_setting(self,req:TargetPose.Request, res: TargetPose.Response) -> TargetPose.Response:
        # self.get_logger().info(f'{req.prepared}')
        if self.status == True:
            plate_tf = self.tf_buffer.lookup_transform('base_plate', 'person_link', rclpy.time.Time())
            res.x = float("{:.3f}".format(plate_tf.transform.translation.x))
            res.y = float("{:.3f}".format(plate_tf.transform.translation.y))
            res.z = float("{:.3f}".format(plate_tf.transform.translation.z))
            res.w = 1.0
            res.status = self.status
            self.get_logger().info('\033[93m Sending: x : {}  y:  {}  z: {} w: 1.0\033[0m'.format(res.x,res.y,res.z)) 
        return res

    def person_setting(self, req: Person.Request, res: Person.Response ) -> Person.Response:
        self.person_name = req.person_name
        res.success_name = self.person_name
        return res
    
    def setXY(self, keypoints_msg: KeyPoint2DArray):
        cnt = 0
        sh_point = [0, 0]
        for kp in keypoints_msg.data:
            #### Shoulder middle point!
            if str(kp.id) == '7' or str(kp.id) == '6':
                cnt+=1
                sh_point[0] += kp.point.x
                sh_point[1] += kp.point.y
        
        if cnt != 0:
            sh_point[0],sh_point[1] = sh_point[0]//cnt,sh_point[1]//cnt
        
        return sh_point[0], sh_point[1]
    
    def person_tf(self, depth_msg: Image, face_detection_msg: DetectionArray):
        point_x = 0
        point_y = 0
        detection: Detection
        # self.get_logger().info('\033[93m ======================={}======================= \033[0m'.format(self.person_name))
        self.status = False
        for detection in face_detection_msg.detections:
            if detection.facebox.name == self.person_name:    
                point_x, point_y = self.setXY(detection.keypoints)
                self.status = True
                break # 이후 프로세스 진행
        
        if point_x == 0:
            self.status = False
            return # 프로세스 종료
        
        depth_frame = self.cv_bridge.imgmsg_to_cv2(depth_msg)

        u = int(point_x)
        v = int(point_y)
        depth = depth_frame[v][u]  # Access depth at (v, u) due to OpenCV array indexing
        
        if depth != 0:
            # Convert pixel coordinates to camera coordinates
            image_center = [depth_msg.width / 2.0, depth_msg.height / 2.0]
            focal_length = 381.98  # Focal length in pixels (example value)
            camera_coords = self.pixel_to_camera_coordinates(u, v, depth, focal_length, image_center)

            # Get transform from camera_link to base_link
            try:
                current_time = self.get_clock().now().to_msg()
                transform = self.tf_buffer.lookup_transform('camera_link', 'camera_color_frame', rclpy.time.Time())
                camera_position = np.array([transform.transform.translation.x,
                                            transform.transform.translation.y,
                                            transform.transform.translation.z])
                camera_orientation = transform.transform.rotation

                # Convert quaternion to rotation matrix
                R = self.quaternion_to_rotation_matrix(camera_orientation)

                # Transform camera coordinates to world coordinates
                object_position_camera_frame = camera_coords
                object_position_world_frame = np.dot(R, object_position_camera_frame) + camera_position

                transform_stamped = TransformStamped()
                transform_stamped.header.stamp = current_time
                transform_stamped.header.frame_id = 'camera_color_frame'
                transform_stamped.child_frame_id = 'person_link'
                
                # self.x = float("{:.3f}".format(object_position_world_frame[2] / 1000.0))
                # self.y =  float("{:.3f}".format(object_position_world_frame[0] / 1000.0))
                # self.z = float("{:.3f}".format(object_position_world_frame[1] / 1000.0))
                
                # For debugging
                transform_stamped.transform.translation.x = float("{:.3f}".format(object_position_world_frame[2] / 1000.0))
                transform_stamped.transform.translation.y = float("{:.3f}".format(object_position_world_frame[0] / 1000.0)) 
                transform_stamped.transform.translation.z = float("{:.3f}".format(object_position_world_frame[1] / 1000.0))
                
                transform_stamped.transform.rotation.x = 0.0
                transform_stamped.transform.rotation.y = 0.240207
                transform_stamped.transform.rotation.z = 0.0
                transform_stamped.transform.rotation.w = 0.74556
                self.person_broadcaster.sendTransform(transform_stamped)
                # self.get_logger().info('\033[93m depth {} : x:{} | y:{} | z:{}\033[0m'.format(depth, self.x, self.y, self.z))

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                self.get_logger().error(f"Failed to lookup transform: {e}")

    def pixel_to_camera_coordinates(self, u, v, depth, focal_length, image_center):
        z_camera = depth
        x_camera = (u - image_center[0]) * depth / focal_length
        y_camera = (v - image_center[1]) * depth / focal_length
        return np.array([x_camera, y_camera, z_camera])

    def quaternion_to_rotation_matrix(self, quaternion):
        q = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        return txq.quat2mat(q)

def main(args=None):
    rclpy.init(args=args)
    try:
        node = WorldNode()
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.\n')
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()