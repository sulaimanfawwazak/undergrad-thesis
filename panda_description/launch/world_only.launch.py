import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node


def generate_launch_description():
    panda_description = get_package_share_directory("panda_description")
    
    world_name_arg = DeclareLaunchArgument(
        name="world_name",
        default_value="test",  # Use your new world file
        description="World file name (without .world extension)"
    )
    
    world_path = PathJoinSubstitution([
        panda_description,
        "worlds",
        PythonExpression(expression=["'", LaunchConfiguration("world_name"), "'", " + '.world'"])
    ])
    
    # Set resource path for models
    model_path = str(Path(panda_description).parent.resolve())
    model_path += ":" + os.path.join(get_package_share_directory("panda_description"), 'models')
    
    gazebo_resource_path = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        model_path
    )
    
    # Start Gazebo with world only (no robot)
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory("ros_gz_sim"), "launch"),
            "/gz_sim.launch.py"
        ]),
        launch_arguments={
            "gz_args": PythonExpression(["'", world_path, " -v 4 -r'"])
        }.items()
    )
    
    return LaunchDescription([
        world_name_arg,
        gazebo_resource_path,
        gazebo,
    ])