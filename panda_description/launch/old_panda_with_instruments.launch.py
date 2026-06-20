# panda_description/launch/panda_with_instruments.launch.py
import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution, Command
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from pathlib import Path
from os import pathsep

def generate_launch_description():
    panda_description = get_package_share_directory("panda_description")
    
    # Arguments
    world_name_arg = DeclareLaunchArgument(
        name="world_name",
        default_value="thesis",
        description="World name (without extension)"
    )

    model_arg = DeclareLaunchArgument(
        name="model",
        default_value=os.path.join(
            panda_description,
            "urdf",
            "panda.urdf.xacro"
        ),
        description="Absolute path to robot urdf file")
    
    randomize_arg = DeclareLaunchArgument(
        "randomize_instruments",
        default_value="true",
        description="Randomize instrument positions"
    )
    
    random_seed_arg = DeclareLaunchArgument(
        "random_seed",
        default_value="42",
        description="Random seed"
    )
    
    # World file path
    world_path = PathJoinSubstitution([
        panda_description,
        "worlds", 
        [LaunchConfiguration("world_name"), TextSubstitution(text=".world")]
    ])

    model_path = str(Path(panda_description).parent.resolve()) # panda_description/share/panda_description -> panda_description/share/
    model_path += pathsep + os.path.join(get_package_share_directory("panda_description"), 'models') # panda_description/share/models

    # Path for meshes, models, etc. for Gazebo --> Enables `model://something`
    gazebo_resource_path = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        model_path
    )

    # ROS2 distro checking
    ros_distro = os.environ["ROS_DISTRO"]
    is_ignition = "true" if ros_distro == "humble" else "False"

    # Transform Xacro to URDF
    robot_description = ParameterValue(
        # The same as `xacro panda.urdf.xacro is_ignition:=True`
        Command([
            "xacro ",
            LaunchConfiguration("model"),
            " is_ignition:=",
            is_ignition
        ]),
        value_type=str # Convert the converted Xacro -> URDF (XML) into string
    )

    # Create Robot State Publisher node
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": True
            }
        ]
    )
    
    # Start Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory("ros_gz_sim"), "launch"),
            "/gz_sim.launch.py"
        ]),
        launch_arguments={
            "gz_args": [
                world_path, TextSubstitution(text=" -v 4 -r")
            ]}.items()
    )
    
    # Spawn robot
    spawn_panda = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-topic", "/robot_description",
            "-world", LaunchConfiguration("world_name"),
            "-name", "panda",
            "-x", "0.0",
            "-y", "0.0",
            "-z", "0.725",
            "-R", "0.0",
            "-P", "0.0",
            "-Y", "0.0",
        ]
    )

    spawn_instruments = Node(
        package="panda_description",
        executable="spawn_instruments.py",
        output="screen",
        emulate_tty=True,
        arguments=[
            ["world_name:=", LaunchConfiguration("world_name")],
            ["randomize:=", LaunchConfiguration("randomize_instruments")],
            ["random_seed:=", LaunchConfiguration("random_seed")],
        ]
    )

    # Wait for Gazebo to be ready, then spawn robot
    # delayed_spawn_panda = RegisterEventHandler(
    #     event_handler=OnProcessExit(
    #         target_action=gazebo,
    #         on_exit=[spawn_panda],
    #     )
    # )

    # # Change from 5 seconds to 10-15 seconds
    # delayed_spawn_panda = TimerAction(
    #     period=10.0,  # Increased from 5.0
    #     actions=[spawn_panda]
    # )

    delayed_spawn_instruments = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_panda,
            on_exit=[spawn_instruments],
        )
    )

    ros_gz_rgb_bridge = Node(
        package="ros_gz_image",
        executable="image_bridge",
        # arguments=["/camera/image_raw"]
        arguments=["/camera/image"]
    )

    ros_gz_depth_bridge = Node(
        package="ros_gz_image",
        executable="image_bridge",
        arguments=["/camera/depth_image"]
    )

    # Bridge for anything but images
    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        ]
    )
    
    return LaunchDescription([
        # Arguments
        world_name_arg,
        model_arg,
        randomize_arg,
        random_seed_arg,

        gazebo_resource_path,
        robot_state_publisher_node,
        gazebo,
        spawn_panda,
        # delayed_spawn_panda,
        # spawn_instruments,
        delayed_spawn_instruments,
        ros_gz_rgb_bridge,
        ros_gz_depth_bridge,
        ros_gz_bridge
    ])