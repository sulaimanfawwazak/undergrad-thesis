# panda_bringup/launch/pick_and_place_yolo.launch.py
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    # ------------------- Gazebo -------------------
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("panda_description"),
                "launch",
                "panda_with_instruments.launch.py"
            )
        ),
        launch_arguments={"randomize_instruments": "false"}.items()
    )

    # ------------------- Controllers -------------------
    controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("panda_controller"),
                "launch",
                "panda_controller.launch.py"
            )
        ),
        launch_arguments={"is_sim": "true"}.items()
    )

    # ------------------- MoveIt -------------------
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("panda_moveit"),
                "launch",
                "moveit.launch.py"
            )
        ),
        launch_arguments={"is_sim": "true"}.items()
    )

    # ------------------- Vision Node -------------------
    vision_node = Node(
        package="panda_vision",
        # executable="color_detector",
        executable="yolo_detector",
        name="yolo_detector",
        output="screen"
    )

    # ------------------- Vision Node -------------------
    # sr_node = Node(
    #     package="panda_vision",
    #     executable="speech_controller",
    #     name="speech_controller",
    #     output="screen"
    # )

    # ------------------- MoveIt Color Picker Node -------------------
    color_picker_node = Node(
        package="pymoveit2",
        executable="pick_and_place.py",
        name="pick_and_place",
        output="screen",
        parameters=[
            {"target_color": "B"}  # {"target_color": "R"}, {"target_color": "G"}
        ]
    )

    return LaunchDescription([
        gazebo,
        controller,
        moveit,
        vision_node,
        # sr_node,
        # color_picker_node,
    ])