#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseArray, TransformStamped, PoseWithCovarianceStamped
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import csv
import tf_transformations
from tf2_ros import TransformBroadcaster
from ament_index_python.packages import get_package_share_directory
from rclpy.duration import Duration

class MarkerWorldInfo:
    def __init__(self, marker_id, x, y, z, roll, pitch, yaw, size_cm, description):
        self.id = int(marker_id)
        self.world_x = float(x)
        self.world_y = float(y)
        self.world_z = float(z)
        self.roll = float(roll)
        self.pitch = float(pitch)
        self.yaw = float(yaw)
        self.size_cm = float(size_cm)
        self.description = str(description)

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')

        # Parameters
        self.declare_parameter('camera_topic', '/camera/realsense_camera/color/image_raw')
        self.declare_parameter('camera_info', '/camera/realsense_camera/color/camera_info')
        self.declare_parameter('database_file', os.path.expanduser('~/agribot_ws/src/agribot_landmarks/markers/marker_database.csv'))
        self.declare_parameter('marker_size', 0.1)  # meters
        self.declare_parameter('camera_frame', 'camera_link_aligned')
        self.declare_parameter('world_frame', 'odom')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('publish_markers', True)
        self.declare_parameter('min_markers_for_localization', 1)
        self.declare_parameter('show_image', True)  # New parameter for visualization

        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info').get_parameter_value().string_value
        self.database_file = self.get_parameter('database_file').get_parameter_value().string_value
        self.marker_size = self.get_parameter('marker_size').get_parameter_value().double_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        self.publish_tf = self.get_parameter('publish_tf').get_parameter_value().bool_value
        self.publish_markers = self.get_parameter('publish_markers').get_parameter_value().bool_value
        self.min_markers = self.get_parameter('min_markers_for_localization').get_parameter_value().integer_value
        self.show_image = self.get_parameter('show_image').get_parameter_value().bool_value


        self.get_logger().info(f"Camera frame parameter: {self.camera_frame}")
        self.get_logger().info(f"World frame parameter: {self.world_frame}")
        
        
        # Camera parameters 
        self.K = None
        self.dist_coeffs = None
        self.camera_info_received = False

        # ArUco
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        
        try:
            self.detector_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)
        except AttributeError:
            # Older OpenCV version
            self.detector_params = cv2.aruco.DetectorParameters_create()
            self.detector = None

        # Camera calibration 
        self.camera_info_sub_ = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info, 10)

        # Marker DB
        self.marker_database = {}
        self.load_marker_database()

        # ROS2 interface
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, self.camera_topic, self.image_callback, 10)
        self.detection_pub = self.create_publisher(PoseArray, '/aruco_detections', 10)
        self.world_pose_pub = self.create_publisher(PoseArray, '/aruco_world_poses', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/aruco_markers', 10)
        self.camera_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/camera_pose', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(f"ArucoDetectorNode initialized, listening to {self.camera_topic}")

    def load_marker_database(self):
        try:
            with open(self.database_file, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    info = MarkerWorldInfo(
                        int(row['ID']),
                        float(row['World_X']), float(row['World_Y']), float(row['World_Z']),
                        float(row['Roll']), float(row['Pitch']), float(row['Yaw']),
                        float(row['Size_CM']), row['Description']
                    )
                    self.marker_database[info.id] = info

                    # Debug: Log loaded marker info
            #         self.get_logger().info(
            #             f"Loaded marker {info.id}: world=({info.world_x}, {info.world_y}, {info.world_z}), "
            #             f"size={info.size_cm}cm, orientation=(r:{info.roll}, p:{info.pitch}, y:{info.yaw})")
            # self.get_logger().info(f"Loaded {len(self.marker_database)} markers from DB")
        except Exception as e:
            self.get_logger().error(f"Failed to load marker database: {e}")

    def camera_info(self, info):
        self.K = np.array(info.k, dtype=np.float64).reshape((3, 3))
        self.dist_coeffs = np.array(info.d, dtype=np.float64) if len(info.d) > 0 else np.zeros(4, dtype=np.float64)
        self.camera_info_received = True
        self.get_logger().info("Camera info received successfully")
        self.destroy_subscription(self.camera_info_sub_)

    # def compute_camera_pose_from_marker(self, marker_id, marker_tvec_cam, marker_rvec_cam):
    #     """
    #     Compute camera pose in world frame from marker detection
    #     """
    #     if marker_id not in self.marker_database:
    #         return None, None
            
    #     marker_info = self.marker_database[marker_id]
        
    #     # Marker pose in world frame
    #     T_marker_world = np.array([marker_info.world_x, marker_info.world_y, marker_info.world_z])
    #     R_marker_world = tf_transformations.euler_matrix(
    #         marker_info.roll, marker_info.pitch, marker_info.yaw, 'rxyz')[:3, :3]
        
    #     # Marker pose relative to camera (from ArUco detection)
    #     T_marker_camera = marker_tvec_cam.reshape(3,)
    #     R_marker_camera, _ = cv2.Rodrigues(marker_rvec_cam)
        
    #     # Camera pose relative to marker
    #     R_camera_marker = R_marker_camera.T
    #     T_camera_marker = -R_camera_marker @ T_marker_camera
        
    #     # Camera pose in world frame
    #     # T_cam_world = T_marker_world * T_cam_marker
    #     R_camera_world = R_marker_world @ R_camera_marker
    #     T_camera_world = T_marker_world + R_marker_world @ T_camera_marker
        
    #     camera_euler = tf_transformations.euler_from_matrix(
    #         np.vstack([np.hstack([R_camera_world, T_camera_world.reshape(-1,1)]), [0,0,0,1]]))
        
    #     # T_cam_world_homogeneous = np.eye(4)
    #     # T_cam_world_homogeneous[:3, :3] = R_camera_world
    #     # T_cam_world_homogeneous[:3, 3] = T_camera_world
    #     # return T_cam_world_homogeneous, camera_euler[:3]
        
    #     return T_camera_world, camera_euler[:3]

    def draw_marker_info(self, frame, corners, ids, rvecs=None, tvecs=None):
        """
        Draw marker detection information on the frame
        """
        if ids is None:
            return frame
        
        # Create a copy to draw on
        display_frame = frame.copy()
        
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
        
        # Draw axes if pose estimation is available
        if rvecs is not None and tvecs is not None and self.camera_info_received:
            for i, marker_id in enumerate(ids.flatten()):
                # Get marker size
                if marker_id in self.marker_database:
                    marker_size_m = self.marker_database[marker_id].size_cm / 100.0
                else:
                    marker_size_m = self.marker_size
                
                # Draw 3D axes
                cv2.drawFrameAxes(display_frame, self.K, self.dist_coeffs, 
                                rvecs[i], tvecs[i], marker_size_m * 0.5)
                
                # Add text with marker info
                corner = corners[i][0]
                center = np.mean(corner, axis=0).astype(int)
                
                # Distance calculation
                distance = np.linalg.norm(tvecs[i][0])
                
                # Marker info text
                # distance_text = f"Dist: {distance:.2f}m"
                
                # # Database info if available
                # if marker_id in self.marker_database:
                #     marker_info = self.marker_database[marker_id]
                #     desc_text = f"Desc: {marker_info.description}"
                #     world_pos_text = f"World: ({marker_info.world_x:.1f}, {marker_info.world_y:.1f}, {marker_info.world_z:.1f})"
                # else:
                #     desc_text = "Unknown marker"
                #     world_pos_text = "No world info"
                
                # # Draw background rectangles for text
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # font_scale = 0.6
                # thickness = 2
                
        return display_frame

    def camera_coordinate_transformation(self, rvec, tvec):
        """
        Transforms the pose from OpenCV camera frame (Z-forward, X-right, Y-down) to aligne robot frame (X-forward, Y-left, Z-up)
        """
        rvec = np.array(rvec).flatten()
        tvec = np.array(tvec).flatten()
        
        R_marker_to_camera, _ = cv2.Rodrigues(rvec)

        R_align = np.array([
            [ 0,  0, 1],  # X_forward = Z_opencv
            [-1,  0, 0],  # Y_left    = -X_opencv
            [ 0, -1, 0]   # Z_up      = -Y_opencv
        ])

        R_aligned = R_align @ R_marker_to_camera
        T_aligned = R_align @ tvec.reshape(3, 1)

        # Convert back to Rodrigues vector
        rvec_aligned, _ = cv2.Rodrigues(R_aligned)

        return rvec_aligned.flatten(), T_aligned.flatten(), R_aligned

    def tf_marker_in_camera_frame(self, R_aligned, tvec_aligned, marker_id):
        quat = tf_transformations.quaternion_from_matrix(np.vstack([np.hstack([R_aligned, np.array([[0], [0], [0]])]),[0, 0, 0, 1]]))
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.camera_frame
        t.child_frame_id = f"aruco_marker_{marker_id}"

        t.transform.translation.x = float(tvec_aligned[0])
        t.transform.translation.y = float(tvec_aligned[1])
        t.transform.translation.z = float(tvec_aligned[2])
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

    def publish_camera_world_transform(self, camera_world_pos, camera_world_quat):
        """
        Publish the camera pose in world frame as a TF transform.
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "odom"  # or your world frame name
        t.child_frame_id = f"{self.camera_frame}"

        t.transform.translation.x = float(camera_world_pos[0])
        t.transform.translation.y = float(camera_world_pos[1])
        t.transform.translation.z = float(camera_world_pos[2])
        t.transform.rotation.x = camera_world_quat[0]
        t.transform.rotation.y = camera_world_quat[1]
        t.transform.rotation.z = camera_world_quat[2]
        t.transform.rotation.w = camera_world_quat[3]

        self.tf_broadcaster.sendTransform(t)

    def estimate_camera_pose_in_world(self, rvec_aligned, tvec_aligned, R_aligned, marker_id):
        """
        Estimated camera pose in world reference frame from marker detection.
        
        Args:
            rvec_aligned: Rotation vector from marker to camera (aligned coordinates)
            tvec_aligned: Translation vector from marker to camera (aligned coordinates)  
            R_aligned: Rotation matrix from marker to camera (aligned coordinates)
            marker_world_pos: [x, y, z] position of marker in world frame
            marker_world_euler: [roll, pitch, yaw] orientation of marker in world frame (radians)
        """
        marker = self.marker_database[marker_id]

        # marker position and orientation in world refernce frame
        marker_world_position = np.array([marker.world_x, marker.world_y, marker.world_z])   # meter
        marker_world_orientation = np.array([marker.roll, marker.pitch, marker.yaw])         # rad
        
        # homogeneous transformation matrix from camera to marker
        T_camera_marker = np.eye(4)
        T_camera_marker[:3, :3] = R_aligned
        T_camera_marker[:3, 3] = tvec_aligned.flatten()
        
        # Invert to get marker to camera transform
        T_marker_camera = np.linalg.inv(T_camera_marker)
        
        # World to marker transformation
        # Convert Euler angles to rotation matrix
        R_world_marker = tf_transformations.euler_matrix(
            marker_world_orientation[0],  
            marker_world_orientation[1],    
            marker_world_orientation[2],  
            'rxyz'  # rotation order
        )[:3, :3]
        
        T_world_marker = np.eye(4)
        T_world_marker[:3, :3] = R_world_marker
        T_world_marker[:3, 3] = marker_world_position
        
        # 4. Calculate camera pose in world frame
        T_world_cam = T_world_marker @ T_marker_camera
        
        # 5. Extract position and orientation
        camera_world_pos = T_world_cam[:3, 3]
        camera_world_rot_matrix = T_world_cam[:3, :3]
        
        # Convert rotation matrix to quaternion
        T_world_cam_homogeneous = np.eye(4)
        T_world_cam_homogeneous[:3, :3] = camera_world_rot_matrix
        camera_world_quat = tf_transformations.quaternion_from_matrix(T_world_cam_homogeneous)
        
        # Convert to Euler angles if needed
        camera_world_euler = tf_transformations.euler_from_matrix(T_world_cam_homogeneous, 'rxyz')
        
        return camera_world_pos, camera_world_quat, camera_world_euler

    def compute_camera_pose_from_marker(self, marker_id, marker_tvec_cam, marker_rvec_cam):
        """
        Compute camera pose in world frame from marker detection
        
        Args:
            marker_id: ID of the detected marker
            marker_tvec_cam: Translation vector of marker in camera frame (already aligned)
            marker_rvec_cam: Rotation vector of marker in camera frame (already aligned)
        
        Returns:
            camera_position_world: Camera position in world coordinates
            camera_euler_world: Camera orientation in world coordinates (roll, pitch, yaw)
        """
        if marker_id not in self.marker_database:
            return None, None
            
        marker_info = self.marker_database[marker_id]

        marker_tvec_cam = np.array(marker_tvec_cam).flatten()
        marker_rvec_cam = np.array(marker_rvec_cam).flatten()
        
        # Step 1: Marker pose in world frame
        T_marker_world = np.array([marker_info.world_x, marker_info.world_y, marker_info.world_z])
        R_marker_world = tf_transformations.euler_matrix(
            marker_info.roll, marker_info.pitch, marker_info.yaw, 'rxyz')[:3, :3]
        
        # Step 2: Marker pose relative to camera (from ArUco detection - already aligned)
        T_marker_camera = marker_tvec_cam.flatten()
        R_marker_camera, _ = cv2.Rodrigues(marker_rvec_cam)
        
        # Step 3: INVERT to get camera pose relative to marker
        # For transformation T = [R|t], inverse is T^-1 = [R^T | -R^T * t]
        R_camera_marker = R_marker_camera.T
        T_camera_marker = -R_camera_marker @ T_marker_camera
        
        # Step 4: Compose transformations to get camera pose in world frame
        # T_camera_world = T_marker_world * T_camera_marker
        R_camera_world = R_marker_world @ R_camera_marker
        T_camera_world = T_marker_world + R_marker_world @ T_camera_marker
        
        # Step 5: Convert rotation matrix to euler angles
        # Create homogeneous transformation matrix for conversion
        T_homogeneous = np.eye(4)
        T_homogeneous[:3, :3] = R_camera_world
        T_homogeneous[:3, 3] = T_camera_world
        
        camera_euler = tf_transformations.euler_from_matrix(T_homogeneous)
        
        return T_camera_world, camera_euler[:3]

    def create_homogeneous_matrix(self, rotation_matrix, translation_vector):
        """
        Create 4x4 homogeneous transformation matrix from rotation and translation
        """
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = translation_vector.flatten()
        return T

    def validate_transformation(self, marker_id, T_camera_world, marker_corners, camera_matrix, dist_coeffs):
        """
        Validate the camera localization by reprojecting marker corners
        """
        if marker_id not in self.marker_database:
            return False, float('inf')
        
        marker_info = self.marker_database[marker_id]
        marker_size = marker_info.size_cm / 100.0  # Convert to meters
        
        # Define marker corners in marker coordinate system
        half_size = marker_size / 2.0
        marker_corners_3d = np.array([
            [-half_size, -half_size, 0],
            [ half_size, -half_size, 0],
            [ half_size,  half_size, 0],
            [-half_size,  half_size, 0]
        ], dtype=np.float32)
        
        # Transform marker corners to world coordinates
        T_marker_world = self.create_homogeneous_matrix(
            tf_transformations.euler_matrix(
                marker_info.roll, marker_info.pitch, marker_info.yaw, 'rxyz')[:3, :3],
            np.array([marker_info.world_x, marker_info.world_y, marker_info.world_z])
        )
        
        # Transform to camera coordinates
        T_world_camera = np.linalg.inv(T_camera_world)
        
        corners_world_homogeneous = np.hstack([marker_corners_3d, np.ones((4, 1))])
        corners_world = (T_marker_world @ corners_world_homogeneous.T).T[:, :3]
        corners_camera = (T_world_camera @ np.hstack([corners_world, np.ones((4, 1))]).T).T[:, :3]
        
        # Project to image plane
        projected_corners, _ = cv2.projectPoints(
            corners_camera, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
        projected_corners = projected_corners.reshape(-1, 2)
        
        # Calculate reprojection error
        detected_corners = marker_corners[0].reshape(-1, 2)
        reprojection_error = np.mean(np.linalg.norm(projected_corners - detected_corners, axis=1))
        
        # Validation: error should be less than a few pixels
        is_valid = reprojection_error < 5.0  # pixels
        
        return is_valid, reprojection_error

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Handle different OpenCV versions for marker detection
            if self.detector is not None:
                corners, ids, _ = self.detector.detectMarkers(frame)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(frame, self.dictionary, parameters=self.detector_params)
                
            rvecs_all = None
            tvecs_all = None
            
            pose_array = PoseArray()
            pose_array.header = msg.header
            pose_array.header.frame_id = self.camera_frame 


            world_pose_array = PoseArray()
            world_pose_array.header = msg.header
            world_pose_array.header.frame_id = self.world_frame

            marker_array = MarkerArray()

            if ids is not None and len(ids) > 0:
                detected_markers = [int(id_val) for id_val in ids.flatten()]
                # self.get_logger().info(f"Detected markers: {detected_markers}")
                
                rvecs_list = []
                tvecs_list = []
                
                # Process individual marker poses (relative to camera)
                for i, marker_id in enumerate(detected_markers):
                    
                    if self.camera_info_received:
                        try:
                            if marker_id in self.marker_database:
                                marker_size_m = self.marker_database[marker_id].size_cm / 100.0
                                self.get_logger().info(f"Using database size for marker {marker_id}: {marker_size_m:.3f}m")
                            else:
                                marker_size_m = self.marker_size  # Fallback to parameter
                                self.get_logger().info(f"Using parameter size for marker {marker_id}: {marker_size_m:.3f}m")
                            
                            # Translation and rotation of marker in camera coordiniate system - computed by PnP
                            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                                [corners[i]], marker_size_m, self.K, self.dist_coeffs)
                 
                            # Transform from camera coordinate (Z-forward, X-right, Y-down) to camera coordinate (X-forward, Y-left, Z-up)                        
                            rvec_aligned, tvec_aligned, R_aligned = self.camera_coordinate_transformation(rvec[0], tvec[0])
                            
                            # tf of marker in camera frame
                            self.tf_marker_in_camera_frame(R_aligned, tvec_aligned, marker_id)

                            cam_pos, cam_quat, cam_euler = self.estimate_camera_pose_in_world(rvec_aligned, tvec_aligned, R_aligned, marker_id)

                            self.publish_camera_world_transform(cam_pos, cam_quat)

                        
                            # # Store for visualization
                            # rvecs_list.append(rvec_aligned)
                            # tvecs_list.append(tvec_aligned)
                            
                            # distance = np.linalg.norm(tvec_aligned)
                            # # self.get_logger().info(
                            # #     f"Marker {marker_id} detected at: "
                            # #     f"x={tvec_aligned[0]:.3f}, y={tvec_aligned[1]:.3f}, z={tvec_aligned[2]:.3f}, "
                            # #     f"distance={distance:.3f}m")
                            
                            # pose = Pose()
                            # pose.position.x = float(tvec_aligned[0])
                            # pose.position.y = float(tvec_aligned[1])
                            # pose.position.z = float(tvec_aligned[2])
                            
                            # quat = tf_transformations.quaternion_from_euler(
                            #     float(rvec_aligned[0]), float(rvec_aligned[1]), float(rvec_aligned[2]))
                            # pose.orientation.x = float(quat[0])
                            # pose.orientation.y = float(quat[1])
                            # pose.orientation.z = float(quat[2])
                            # pose.orientation.w = float(quat[3])
                            # pose_array.poses.append(pose)
                            
                            # if marker_id in self.marker_database:
                            #     cam_pos_world, cam_rot_world = self.compute_camera_pose_from_marker(
                            #         marker_id, tvec_aligned, rvec_aligned)
                                
                            #     if cam_pos_world is not None:
                            #         # Validate the transformation
                            #         is_valid, reprojection_error = self.validate_transformation(
                            #             marker_id, self.create_homogeneous_matrix(
                            #                 tf_transformations.euler_matrix(
                            #                     cam_rot_world[0], cam_rot_world[1], cam_rot_world[2], 'rxyz')[:3, :3],
                            #                 cam_pos_world), 
                            #             corners[i], self.K, self.dist_coeffs)
                                    
                            #         self.get_logger().info(
                            #             f"Camera localization validation - Valid: {is_valid}, "
                            #             f"Reprojection error: {reprojection_error:.2f} pixels")
                                    
                            #         if is_valid:  # Only publish if validation passes
                            #             # Publish camera pose
                            #             camera_pose_msg = PoseWithCovarianceStamped()
                            #             camera_pose_msg.header.stamp = msg.header.stamp
                            #             camera_pose_msg.header.frame_id = self.world_frame
                                        
                            #             camera_pose_msg.pose.pose.position.x = float(cam_pos_world[0])
                            #             camera_pose_msg.pose.pose.position.y = float(cam_pos_world[1])
                            #             camera_pose_msg.pose.pose.position.z = float(cam_pos_world[2])
                                        
                            #             quat_cam = tf_transformations.quaternion_from_euler(
                            #                 cam_rot_world[0], cam_rot_world[1], cam_rot_world[2])
                            #             camera_pose_msg.pose.pose.orientation.x = float(quat_cam[0])
                            #             camera_pose_msg.pose.pose.orientation.y = float(quat_cam[1])
                            #             camera_pose_msg.pose.pose.orientation.z = float(quat_cam[2])
                            #             camera_pose_msg.pose.pose.orientation.w = float(quat_cam[3])
                                        
                            #             # Covariance based on reprojection error
                            #             base_covariance = 0.01 + (reprojection_error / 100.0)  # Scale with error
                            #             camera_pose_msg.pose.covariance = [base_covariance] * 36
                                        
                            #             self.camera_pose_pub.publish(camera_pose_msg)
                                                    
                            #             # Publish TF for camera localization
                            #             if self.publish_tf:
                            #                 t = TransformStamped()
                            #                 t.header.stamp = msg.header.stamp
                            #                 t.header.frame_id = self.world_frame
                            #                 t.child_frame_id = f"{self.camera_frame}"
                            #                 t.transform.translation.x = float(cam_pos_world[0])
                            #                 t.transform.translation.y = float(cam_pos_world[1])
                            #                 t.transform.translation.z = float(cam_pos_world[2])
                            #                 t.transform.rotation.x = float(quat_cam[0])
                            #                 t.transform.rotation.y = float(quat_cam[1])
                            #                 t.transform.rotation.z = float(quat_cam[2])
                            #                 t.transform.rotation.w = float(quat_cam[3])
                            #                 self.tf_broadcaster.sendTransform(t)
                                        
                            #             self.get_logger().info(
                            #                 f"Camera localized using marker {marker_id} at "
                            #                 f"({cam_pos_world[0]:.2f}, {cam_pos_world[1]:.2f}, {cam_pos_world[2]:.2f})")
                                    # else:
                                    #     self.get_logger().warn(f"Camera localization failed validation for marker {marker_id}")
                        
                        except Exception as e:
                            self.get_logger().error(f"Error processing marker {marker_id}: {e}")
                           
                            rvecs_list.append(None)
                            tvecs_list.append(None)
                    else:
                        rvecs_list.append(None)
                        tvecs_list.append(None)

                    if marker_id in self.marker_database:
                        m = self.marker_database[marker_id]
                        world_pose = Pose()
                        world_pose.position.x = m.world_x
                        world_pose.position.y = m.world_y
                        world_pose.position.z = m.world_z
                        
                        quat = tf_transformations.quaternion_from_euler(m.roll, m.pitch, m.yaw)
                        world_pose.orientation.x = float(quat[0])
                        world_pose.orientation.y = float(quat[1])
                        world_pose.orientation.z = float(quat[2])
                        world_pose.orientation.w = float(quat[3])
                        world_pose_array.poses.append(world_pose)

                        if self.publish_markers:
                            marker = Marker()
                            marker.header.frame_id = self.world_frame
                            marker.header.stamp = msg.header.stamp
                            marker.id = marker_id 
                            marker.type = Marker.CUBE
                            marker.action = Marker.ADD
                            marker.pose = world_pose
                            marker.scale.x = m.size_cm / 100.0  
                            marker.scale.y = m.size_cm / 100.0  
                            marker.scale.z = 0.01
                            marker.color.r = 0.0
                            marker.color.g = 1.0
                            marker.color.b = 0.0
                            marker.color.a = 0.8
                            marker.lifetime = Duration(seconds=1).to_msg()
                            marker_array.markers.append(marker)

                if any(rv is not None for rv in rvecs_list):
                    rvecs_all = np.array([rv for rv in rvecs_list if rv is not None])
                    tvecs_all = np.array([tv for tv in tvecs_list if tv is not None])

                if self.publish_tf:
                    for marker_id in detected_markers:
                        if marker_id in self.marker_database:
                            m = self.marker_database[marker_id]
                            t = TransformStamped()
                            t.header.stamp = msg.header.stamp
                            t.header.frame_id = self.world_frame
                            t.child_frame_id = f"aruco_{marker_id}"
                            t.transform.translation.x = m.world_x
                            t.transform.translation.y = m.world_y
                            t.transform.translation.z = m.world_z
                            quat = tf_transformations.quaternion_from_euler(m.roll, m.pitch, m.yaw)
                            t.transform.rotation.x = float(quat[0])
                            t.transform.rotation.y = float(quat[1])
                            t.transform.rotation.z = float(quat[2])
                            t.transform.rotation.w = float(quat[3])
                            self.tf_broadcaster.sendTransform(t)

            # Publish all pose arrays and markers
            self.detection_pub.publish(pose_array)
            self.world_pose_pub.publish(world_pose_array)
            if self.publish_markers:
                self.marker_pub.publish(marker_array)

            # Display the image with markers drawn
            if self.show_image:
                display_frame = self.draw_marker_info(frame, corners, ids, rvecs_all, tvecs_all)
                cv2.imshow("ArUco Detector", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.get_logger().info("Quit key pressed, shutting down...")
                    rclpy.shutdown()
          
        except Exception as e:
            self.get_logger().error(f"Image callback failed: {e}")
            import traceback
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received")
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()







































# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image, CameraInfo
# from geometry_msgs.msg import Pose, PoseArray, TransformStamped, PoseWithCovarianceStamped
# from std_msgs.msg import Header
# from visualization_msgs.msg import Marker, MarkerArray
# from cv_bridge import CvBridge
# import cv2
# import numpy as np
# import os
# import csv
# import tf_transformations
# from tf2_ros import TransformBroadcaster
# from ament_index_python.packages import get_package_share_directory
# from rclpy.duration import Duration

# class MarkerWorldInfo:
#     def __init__(self, marker_id, x, y, z, roll, pitch, yaw, size_cm, description):
#         self.id = int(marker_id)
#         self.world_x = float(x)
#         self.world_y = float(y)
#         self.world_z = float(z)
#         self.roll = float(roll)
#         self.pitch = float(pitch)
#         self.yaw = float(yaw)
#         self.size_cm = float(size_cm)
#         self.description = str(description)

# class ArucoDetector(Node):
#     def __init__(self):
#         super().__init__('aruco_detector')

#         # Parameters
#         self.declare_parameter('camera_topic', '/camera/realsense_camera/color/image_raw')
#         self.declare_parameter('camera_info', '/camera/realsense_camera/color/camera_info')
#         self.declare_parameter('database_file', os.path.expanduser('~/agribot_ws/src/agribot_landmarks/markers/marker_database.csv'))
#         self.declare_parameter('marker_size', 0.1)  # meters
#         self.declare_parameter('camera_frame', 'camera_link')
#         self.declare_parameter('world_frame', 'odom')
#         self.declare_parameter('publish_tf', True)
#         self.declare_parameter('publish_markers', True)
#         self.declare_parameter('min_markers_for_localization', 1)

#         self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
#         self.camera_info_topic = self.get_parameter('camera_info').get_parameter_value().string_value
#         self.database_file = self.get_parameter('database_file').get_parameter_value().string_value
#         self.marker_size = self.get_parameter('marker_size').get_parameter_value().double_value
#         self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
#         self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
#         self.publish_tf = self.get_parameter('publish_tf').get_parameter_value().bool_value
#         self.publish_markers = self.get_parameter('publish_markers').get_parameter_value().bool_value
#         self.min_markers = self.get_parameter('min_markers_for_localization').get_parameter_value().integer_value

#         # Camera parameters 
#         self.K = None
#         self.dist_coeffs = None
#         self.camera_info_received = False

#         # ArUco
#         self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        
#         try:
#             self.detector_params = cv2.aruco.DetectorParameters()
#             self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)
#         except AttributeError:
#             # Older OpenCV version
#             self.detector_params = cv2.aruco.DetectorParameters_create()
#             self.detector = None

#         # Camera calibration 
#         self.camera_info_sub_ = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info, 10)

#         # Marker DB
#         self.marker_database = {}
#         self.load_marker_database()

#         # ROS2 interface
#         self.bridge = CvBridge()
#         self.image_sub = self.create_subscription(Image, self.camera_topic, self.image_callback, 10)
#         self.detection_pub = self.create_publisher(PoseArray, '/aruco_detections', 10)
#         self.world_pose_pub = self.create_publisher(PoseArray, '/aruco_world_poses', 10)
#         self.marker_pub = self.create_publisher(MarkerArray, '/aruco_markers', 10)
#         self.camera_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/camera_pose', 10)
#         self.tf_broadcaster = TransformBroadcaster(self)

#         self.get_logger().info(f"ArucoDetectorNode initialized, listening to {self.camera_topic}")

#     def load_marker_database(self):
#         try:
#             with open(self.database_file, newline='') as csvfile:
#                 reader = csv.DictReader(csvfile)
#                 for row in reader:
#                     info = MarkerWorldInfo(
#                         int(row['ID']),
#                         float(row['World_X']), float(row['World_Y']), float(row['World_Z']),
#                         float(row['Roll']), float(row['Pitch']), float(row['Yaw']),
#                         float(row['Size_CM']), row['Description']
#                     )
#                     self.marker_database[info.id] = info
#                     # Debug: Log loaded marker info
#                     self.get_logger().info(
#                         f"Loaded marker {info.id}: world=({info.world_x}, {info.world_y}, {info.world_z}), "
#                         f"size={info.size_cm}cm, orientation=(r:{info.roll}, p:{info.pitch}, y:{info.yaw})")
#             self.get_logger().info(f"Loaded {len(self.marker_database)} markers from DB")
#         except Exception as e:
#             self.get_logger().error(f"Failed to load marker database: {e}")

#     def camera_info(self, info):
#         self.K = np.array(info.k, dtype=np.float64).reshape((3, 3))
#         self.dist_coeffs = np.array(info.d, dtype=np.float64) if len(info.d) > 0 else np.zeros(4, dtype=np.float64)
#         self.camera_info_received = True
#         self.get_logger().info("Camera info received successfully")
#         self.destroy_subscription(self.camera_info_sub_)

#     def compute_camera_pose_from_marker(self, marker_id, marker_tvec_cam, marker_rvec_cam):
#         """
#         Compute camera pose in world frame from marker detection
#         """
#         if marker_id not in self.marker_database:
#             return None, None
            
#         marker_info = self.marker_database[marker_id]
        
#         # Marker pose in world frame
#         marker_world_pos = np.array([marker_info.world_x, marker_info.world_y, marker_info.world_z])
        
#         # Marker orientation in world frame (rotation matrix)
#         R_marker_world = tf_transformations.euler_matrix(
#             marker_info.roll, marker_info.pitch, marker_info.yaw, 'rxyz')[:3, :3]
        
#         # Marker pose relative to camera (from ArUco detection)
#         marker_pos_cam = marker_tvec_cam.flatten()
        
#         # Convert marker rotation vector to rotation matrix
#         R_marker_cam, _ = cv2.Rodrigues(marker_rvec_cam)
        
#         # Camera pose relative to marker
#         # T_cam_marker = inverse of T_marker_cam
#         R_cam_marker = R_marker_cam.T
#         t_cam_marker = -R_cam_marker @ marker_pos_cam
        
#         # Camera pose in world frame
#         # T_cam_world = T_marker_world * T_cam_marker
#         R_cam_world = R_marker_world @ R_cam_marker
#         t_cam_world = marker_world_pos + R_marker_world @ t_cam_marker
        
#         # Convert rotation matrix to euler angles
#         camera_euler = tf_transformations.euler_from_matrix(
#             np.vstack([np.hstack([R_cam_world, t_cam_world.reshape(-1,1)]), [0,0,0,1]]))
        
#         return t_cam_world, camera_euler[:3]

#     def image_callback(self, msg):
#         try:
#             frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
#             # Handle different OpenCV versions for marker detection
#             if self.detector is not None:
#                 corners, ids, _ = self.detector.detectMarkers(frame)
#             else:
#                 corners, ids, _ = cv2.aruco.detectMarkers(frame, self.dictionary, parameters=self.detector_params)

#             self.get_logger().info(f"{corners}")
#             # Initialize pose arrays
#             pose_array = PoseArray()
#             pose_array.header = msg.header
#             pose_array.header.frame_id = self.camera_frame  

#             world_pose_array = PoseArray()
#             world_pose_array.header = msg.header
#             world_pose_array.header.frame_id = self.world_frame

#             marker_array = MarkerArray()

#             if ids is not None and len(ids) > 0:

#                 detected_markers = [int(id_val) for id_val in ids.flatten()]
#                 self.get_logger().info(f"Detected markers: {detected_markers}")
                
#                 # Process individual marker poses (relative to camera)
#                 for i, marker_id in enumerate(detected_markers):
                    
#                     if self.camera_info_received:
#                         try:
#                             if marker_id in self.marker_database:
#                                 marker_size_m = self.marker_database[marker_id].size_cm / 100.0
#                                 self.get_logger().info(f"Using database size for marker {marker_id}: {marker_size_m:.3f}m")
#                             else:
#                                 marker_size_m = self.marker_size  # Fallback to parameter
#                                 self.get_logger().info(f"Using parameter size for marker {marker_id}: {marker_size_m:.3f}m")
                            
#                             rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
#                                 [corners[i]], marker_size_m, self.K, self.dist_coeffs)
                            
#                             distance = np.linalg.norm(tvec[0][0])
#                             self.get_logger().info(
#                                 f"Marker {marker_id} detected at: "
#                                 f"x={tvec[0][0][0]:.3f}, y={tvec[0][0][1]:.3f}, z={tvec[0][0][2]:.3f}, "
#                                 f"distance={distance:.3f}m")
                            
#                             pose = Pose()
#                             pose.position.x = float(tvec[0][0][0])
#                             pose.position.y = float(tvec[0][0][1])
#                             pose.position.z = float(tvec[0][0][2])
                            
#                             quat = tf_transformations.quaternion_from_euler(
#                                 float(rvec[0][0][0]), float(rvec[0][0][1]), float(rvec[0][0][2]))
#                             pose.orientation.x = float(quat[0])
#                             pose.orientation.y = float(quat[1])
#                             pose.orientation.z = float(quat[2])
#                             pose.orientation.w = float(quat[3])
#                             pose_array.poses.append(pose)
                            
#                             if marker_id in self.marker_database:
#                                 cam_pos_world, cam_rot_world = self.compute_camera_pose_from_marker(
#                                     marker_id, tvec[0], rvec[0])
                                
#                                 if cam_pos_world is not None:
#                                     # Publish camera pose
#                                     camera_pose_msg = PoseWithCovarianceStamped()
#                                     camera_pose_msg.header.stamp = msg.header.stamp
#                                     camera_pose_msg.header.frame_id = self.world_frame
                                    
#                                     camera_pose_msg.pose.pose.position.x = float(cam_pos_world[0])
#                                     camera_pose_msg.pose.pose.position.y = float(cam_pos_world[1])
#                                     camera_pose_msg.pose.pose.position.z = float(cam_pos_world[2])
                                    
#                                     quat_cam = tf_transformations.quaternion_from_euler(
#                                         cam_rot_world[0], cam_rot_world[1], cam_rot_world[2])
#                                     camera_pose_msg.pose.pose.orientation.x = float(quat_cam[0])
#                                     camera_pose_msg.pose.pose.orientation.y = float(quat_cam[1])
#                                     camera_pose_msg.pose.pose.orientation.z = float(quat_cam[2])
#                                     camera_pose_msg.pose.pose.orientation.w = float(quat_cam[3])
                                    
#                                     # Simple covariance (placeholder values)
#                                     camera_pose_msg.pose.covariance = [0.1] * 36
                                    
#                                     self.camera_pose_pub.publish(camera_pose_msg)
                                    
#                                     # Publish TF for camera localization
#                                     if self.publish_tf:
#                                         t = TransformStamped()
#                                         t.header.stamp = msg.header.stamp
#                                         t.header.frame_id = self.world_frame
#                                         t.child_frame_id = f"{self.camera_frame}_localized"
#                                         t.transform.translation.x = float(cam_pos_world[0])
#                                         t.transform.translation.y = float(cam_pos_world[1])
#                                         t.transform.translation.z = float(cam_pos_world[2])
#                                         t.transform.rotation.x = float(quat_cam[0])
#                                         t.transform.rotation.y = float(quat_cam[1])
#                                         t.transform.rotation.z = float(quat_cam[2])
#                                         t.transform.rotation.w = float(quat_cam[3])
#                                         self.tf_broadcaster.sendTransform(t)
                                    
#                                     self.get_logger().info(
#                                         f"Camera localized using marker {marker_id} at "
#                                         f"({cam_pos_world[0]:.2f}, {cam_pos_world[1]:.2f}, {cam_pos_world[2]:.2f})")
                                        
#                         except Exception as e:
#                             self.get_logger().error(f"Error processing marker {marker_id}: {e}")

#                     if marker_id in self.marker_database:
#                         m = self.marker_database[marker_id]
#                         world_pose = Pose()
#                         world_pose.position.x = m.world_x
#                         world_pose.position.y = m.world_y
#                         world_pose.position.z = m.world_z
                        
                        
#                         quat = tf_transformations.quaternion_from_euler(m.roll, m.pitch, m.yaw)
#                         world_pose.orientation.x = float(quat[0])
#                         world_pose.orientation.y = float(quat[1])
#                         world_pose.orientation.z = float(quat[2])
#                         world_pose.orientation.w = float(quat[3])
#                         world_pose_array.poses.append(world_pose)

#                         if self.publish_markers:
#                             marker = Marker()
#                             marker.header.frame_id = self.world_frame
#                             marker.header.stamp = msg.header.stamp
#                             marker.id = marker_id 
#                             marker.type = Marker.CUBE
#                             marker.action = Marker.ADD
#                             marker.pose = world_pose
#                             marker.scale.x = m.size_cm / 100.0  
#                             marker.scale.y = m.size_cm / 100.0  
#                             marker.scale.z = 0.01
#                             marker.color.r = 0.0
#                             marker.color.g = 1.0
#                             marker.color.b = 0.0
#                             marker.color.a = 0.8
#                             marker.lifetime = Duration(seconds=1).to_msg()
#                             marker_array.markers.append(marker)

#                 # Publish individual marker TFs
#                 if self.publish_tf:
#                     for marker_id in detected_markers:
#                         if marker_id in self.marker_database:
#                             m = self.marker_database[marker_id]
#                             t = TransformStamped()
#                             t.header.stamp = msg.header.stamp
#                             t.header.frame_id = self.world_frame
#                             t.child_frame_id = f"aruco_{marker_id}"
#                             t.transform.translation.x = m.world_x
#                             t.transform.translation.y = m.world_y
#                             t.transform.translation.z = m.world_z
#                             quat = tf_transformations.quaternion_from_euler(m.roll, m.pitch, m.yaw)
#                             t.transform.rotation.x = float(quat[0])
#                             t.transform.rotation.y = float(quat[1])
#                             t.transform.rotation.z = float(quat[2])
#                             t.transform.rotation.w = float(quat[3])
#                             self.tf_broadcaster.sendTransform(t)

#             # Publish all pose arrays and markers
#             self.detection_pub.publish(pose_array)
#             self.world_pose_pub.publish(world_pose_array)
#             if self.publish_markers:
#                 self.marker_pub.publish(marker_array)


#             cv2.imshow("frame", frame)
#             cv2.waitKey(1)


#         except Exception as e:
#             self.get_logger().error(f"Image callback failed: {e}")
#             import traceback
#             self.get_logger().error(f"Traceback: {traceback.format_exc()}")

# def main(args=None):
#     rclpy.init(args=args)
#     node = ArucoDetector()
#     rclpy.spin(node)
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()



