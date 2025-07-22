#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    pkg_dir = get_package_share_directory('agribot_landmarks')
    
    # Declare launch arguments
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/realsense_camera/color/image_raw',
        description='Camera topic name'
    )
    
    database_file_arg = DeclareLaunchArgument(
        'database_file',
        default_value='/home/volcani/agribot_ws/src/agribot_landmarks/markers/marker_database.csv',
        description='Path to marker database file'
    )
    
    marker_size_arg = DeclareLaunchArgument(
        'marker_size',
        default_value='0.1',
        description='Marker size in meters'
    )
    
    camera_frame_arg = DeclareLaunchArgument(
        'camera_frame',
        default_value='camera_link',
        description='Camera frame name'
    )
    
    world_frame_arg = DeclareLaunchArgument(
        'world_frame',
        default_value='map',
        description='World frame name'
    )
    
    publish_tf_arg = DeclareLaunchArgument(
        'publish_tf',
        default_value='true',
        description='Publish TF transforms'
    )
    
    publish_markers_arg = DeclareLaunchArgument(
        'publish_markers',
        default_value='true',
        description='Publish visualization markers'
    )
    
    # Camera node
    camera_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="realsense_camera",
        output="screen",
        parameters=[
            {
            # Camera parameters
            'depth_fps': 30.0,
            # Enable IMU
            'enable_gyro': True,
            'enable_accel': True,
            'gyro_fps': 200.0,   # IMU frequency
            'accel_fps': 250.0,   # accelerometer frequency
            'unite_imu_method': 2, # 0=None , 1=copy, 2=linear_interpolation
            # Synchronization
            'enable_sync': True,
            'align_depth_enable': False,
            'enable_depth': False,
            'enable_infra1': False,
            'enable_infra2': False,
            }],
        )
            




    # ArUco detector node
    aruco_detector_node = Node(
        package='agribot_landmarks',
        executable='aruco_detector.py',
        name='aruco_detector',
        output='screen',
        parameters=[{
            'camera_topic': LaunchConfiguration('camera_topic'),
            'database_file': LaunchConfiguration('database_file'),
            'marker_size': LaunchConfiguration('marker_size'),
            'camera_frame': LaunchConfiguration('camera_frame'),
            'world_frame': LaunchConfiguration('world_frame'),
            'publish_tf': LaunchConfiguration('publish_tf'),
            'publish_markers': LaunchConfiguration('publish_markers'),
            'detection_rate': 10.0
        }]
    )
    
    return LaunchDescription([
        camera_topic_arg,
        database_file_arg,
        marker_size_arg,
        camera_frame_arg,
        world_frame_arg,
        publish_tf_arg,
        publish_markers_arg,
        camera_node,
        aruco_detector_node,
    ])