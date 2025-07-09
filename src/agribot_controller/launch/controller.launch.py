import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.substitutions import LaunchConfiguration, Command
from launch.conditions import IfCondition, UnlessCondition
from ament_index_python.packages import get_package_share_directory
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    #   DISPLAY IN RVIZ

    agribot_description_dir = get_package_share_directory("agribot_description")

    model_arg = DeclareLaunchArgument(
        name="model",
        default_value=os.path.join(agribot_description_dir, "urdf", "agribot.urdf.xacro"),
        description="Absolute path to robot URDF file"
    )    

    robot_description = ParameterValue(Command(["xacro ", LaunchConfiguration("model")]), value_type=str)

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description}]
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", os.path.join(get_package_share_directory("agribot_description"), "rviz", "display.rviz")]
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager"
        ],
    )


    wheel_controller_spwaner =  Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "agribot_controller",
            "--controller-manager",
            "/controller_manager"
        ],
    )


    # simple_controller = Node(
    #     package="controller_manager",
    #     executable="spawner",
    #     arguments=[
    #         "simple_velocity_controller",
    #         "--controller-manager",
    #         "/controller_manager",
    #     ],
    # )





    return LaunchDescription([
        model_arg,
        robot_state_publisher,
        rviz_node,
        joint_state_broadcaster_spawner,
        wheel_controller_spwaner,
        # simple_controller,
        
    ])