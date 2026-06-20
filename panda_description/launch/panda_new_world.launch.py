import os
from os import pathsep
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    panda_description = get_package_share_directory("panda_description")

    model_arg = DeclareLaunchArgument(
        name="model",
        default_value=os.path.join(panda_description, "urdf", "panda.urdf.xacro"),
        description="Absolute path to robot urdf file"
    )

    world_name_arg = DeclareLaunchArgument(
        name="world_name",
        default_value="empty",
        description="World file name (without extension)"
    )

    world_path = PathJoinSubstitution([
            panda_description,
            "worlds",
            PythonExpression(
                expression=["'", LaunchConfiguration("world_name"), "'", " + '.world'"]
            )
        ]
    )

    # Set resource path for models
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

    # Transform Xacro to URDF for robot
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
                    os.path.join(get_package_share_directory("ros_gz_sim"),"launch"),
                    "/gz_sim.launch.py"
                ]),
                launch_arguments={
                    "gz_args": PythonExpression(["'", world_path, " -v 4 -r'"]) # Load world, verbose level 4, run immediately
                }.items()
            )

    # Spawn Panda robot into Gazebo
    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-topic", "robot_description", # Reads from /robot_description topic
            "-name", "panda",
            # Use this instead, let the <origin> in the `arm.xacro` to "0 0 0"
            "-x", "0.0",  
            "-y", "0.0",  
            "-z", "0.725",  
            "-R", "0.0", 
            "-P", "0.0",
            "-Y", "0.0", # Yaw (in radians, e.g., 1.57 for 90 degrees)
        ],
    )

    # Gazebo <-> ROS bridge for: clock & camera
    # gz_ros2_bridge = Node(
    #     package="ros_gz_bridge",
    #     executable="parameter_bridge",
    #     arguments=[
    #         "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
    #         "/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"
    #     ],
    # )

    # Gazebo Image <-> ROS Image
    # ros_gz_image_bridge = Node(
    #     package="ros_gz_image",
    #     executable="image_bridge",
    #     arguments=["/camera/image_raw"]
    # )

    # `ros_gz_image` -> buat image streams
    # `ros_gz_bridge` -> buat non-image data (point cloud, camara info, etc)

    ros_gz_rgb_bridge = Node(
        package="ros_gz_image",
        executable="image_bridge",
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
        model_arg,
        world_name_arg,
        gazebo_resource_path,
        robot_state_publisher_node,
        gazebo,
        gz_spawn_entity,
        # gz_ros2_bridge,
        ros_gz_rgb_bridge, # Comment for now
        ros_gz_depth_bridge, # Comment for now
        ros_gz_bridge # Comment for now
    ])