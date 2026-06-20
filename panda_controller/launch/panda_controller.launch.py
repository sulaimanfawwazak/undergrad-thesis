import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, LaunchConfiguration
from launch.conditions import UnlessCondition
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    is_sim = LaunchConfiguration("is_sim")
    
    is_sim_arg = DeclareLaunchArgument(
        "is_sim",
        default_value="true"
    )

    robot_description = ParameterValue(
        Command(
            [
                "xacro ",
                os.path.join(
                    get_package_share_directory("panda_description"),
                    "urdf",
                    "panda.urdf.xacro",
                ),
                # f" is_sim:=True", # Gabisa pake f-string bolo
                " is_sim:=", is_sim, # Harus pake kek gini
                " is_ignition:=true" # remember to make it according to the Gazebo version
            ]
        ),
        value_type=str,
    )

    # Robot State Publisher node, publishes:
    # - TF tree
    # - frame transforms
    # - robot geometry
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": is_sim
        }],
    )

    # Controller Manager node, loads:
    # - hardware interfaces
    # - joint handles
    # - trajectory controllers
    # - state broadcasters
    # - gripper controller
    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": is_sim
            },
            os.path.join(
                get_package_share_directory("panda_controller"),
                "config",
                "panda_controllers.yaml",
            ),
        ],
    )

    # Joint State Broadcaster node
    # Pubslihes into /joint_states
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
    )

    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller", "--controller-manager", "/controller_manager"],
    )

    return LaunchDescription(
        [
            is_sim_arg,
            robot_state_publisher_node,
            controller_manager,
            joint_state_broadcaster_spawner,
            arm_controller_spawner,
            gripper_controller_spawner,
        ]
    )