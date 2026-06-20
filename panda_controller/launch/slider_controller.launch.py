import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Don't include the full controller since ros2_control is already running from Gazebo (via ign_ros2_control)
    # controller = IncludeLaunchDescription(
    #         os.path.join(
    #             get_package_share_directory("panda_controller"),
    #             "launch",
    #             "panda_controller.launch.py"
    #         ),
    #         launch_arguments={"is_sim": "True"}.items()
    #     )

    joint_state_publisher_gui_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        remappings=[
            ("/joint_states", "/joint_commands"),
        ],
        parameters=[{"use_sim_time": True}]
    )

    slider_control_node = Node(
        package="panda_controller",
        executable="slider_controller.py",
        parameters=[{"use_sim_time": True}]
    )

    return LaunchDescription(
        [
            # controller,  # Commented out since ros2_control is already running
            joint_state_publisher_gui_node,
            slider_control_node
        ]
    )