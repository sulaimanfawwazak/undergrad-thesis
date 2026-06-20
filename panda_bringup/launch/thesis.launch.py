# panda_bringup/launch/final.launch.py
import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, TimerAction, ExecuteProcess, Shutdown, LogInfo
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
    panda_vision = get_package_share_directory("panda_vision")
    pymoveit2 = get_package_share_directory("pymoveit2")
    
    # Arguments
    world_name_arg = DeclareLaunchArgument(name="world_name", default_value="thesis", description="World name (without extension)")
    model_arg = DeclareLaunchArgument(name="model", default_value=os.path.join(panda_description, "urdf", "panda.urdf.xacro"), description="Absolute path to robot urdf file")
    randomize_arg = DeclareLaunchArgument(name="randomize_instruments", default_value="true", description="Randomize instrument positions")
    random_seed_arg = DeclareLaunchArgument(name="random_seed", default_value="42", description="Random seed")
    stt_mode_arg = DeclareLaunchArgument('stt_mode', default_value='mock')
    llm_mode_arg = DeclareLaunchArgument('llm_mode', default_value='mock')
    whisper_model_arg = DeclareLaunchArgument('whisper_model', default_value='base')
    whisper_device_arg = DeclareLaunchArgument('whisper_device', default_value='cpu')
    gemini_model_arg = DeclareLaunchArgument('gemini_model', default_value='gemini-3.1-flash-lite')

    workspace_dir = '/home/pwnwas/Personal/College/Skripsi/Code/playground_ws'
    
    # World file path
    world_path = PathJoinSubstitution([
        panda_description,
        "worlds", 
        [LaunchConfiguration("world_name"), TextSubstitution(text=".world")]
    ])

    # Models path
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

    # ------------------- External Processes -------------------

    # ------------------- YOLO Server -------------------
    yolo_server = ExecuteProcess(
        cmd=[
            '/home/pwnwas/miniconda3/envs/yolo/bin/python',
            f'{workspace_dir}/src/panda_vision/../outside/yolo_server.py',
        ],
        output='screen',
        name='yolo_server',
        on_exit=Shutdown()
    )

    # ------------------- STT Server -------------------
    stt_server = ExecuteProcess(
        cmd=[
            '/home/pwnwas/miniconda3/envs/yolo/bin/python',
            f'{workspace_dir}/src/panda_vision/../outside/stt_server.py',
            '--mode', LaunchConfiguration("stt_mode"),
            '--model', LaunchConfiguration("whisper_model"),
            '--device', LaunchConfiguration("whisper_device"),
        ],
        output='screen',
        name='stt_server',
        on_exit=Shutdown()
    )

    # ------------------- panda_with_instruments.launch.py -------------------

    # ------------------- Transform Xacro to URDF -------------------
    robot_description = ParameterValue(
        # The same as `xacro panda.urdf.xacro is_ignition:=True`
        Command([
            "xacro ", LaunchConfiguration("model"),
            " is_ignition:=", is_ignition
        ]),
        value_type=str # Convert the converted Xacro -> URDF (XML) into string
    )

    # ------------------- Robot State Publisher -------------------
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
    
    # ------------------- Start Gazebo -------------------
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
    
    # ------------------- Spawn Panda -------------------
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

    # ------------------- Spawn Instruments -------------------
    spawn_instruments = Node(
        package="panda_description",
        executable="spawn_instruments.py",
        output="screen",
        emulate_tty=True,
        arguments=[
            ["world_name:=", LaunchConfiguration("world_name")],
            ["randomize:=", LaunchConfiguration("randomize_instruments")],
            ["random_seed:=", LaunchConfiguration("random_seed")]
        ]
    )

    delayed_spawn_instruments = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_panda,
            on_exit=[spawn_instruments]
        )
    )

    # ------------------- Communication Bridges -------------------
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

    # ------------------- surgical_pipeline.launch.py -------------------

    # ------------------- YOLO Detector -------------------
    yolo_node = Node(
        package="panda_vision",
        executable="yolo_detector",
        name="yolo_detector",
        output="screen"
    )

    delayed_yolo_node = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_instruments,
            on_exit=[yolo_node]
        )
    )

    # ------------------- LLM Controller -------------------
    llm_controller = Node(
        package='panda_vision',
        executable='llm_controller',
        name='llm_controller',
        output='screen',
        arguments=[
            '--mode', LaunchConfiguration("llm_mode"),
            '--model', LaunchConfiguration("gemini_model")
        ]
    )

    # ------------------- Instrument Orchestrator -------------------
    orchestrator = Node(
        package='panda_vision',
        executable='instrument_orchestrator',
        name='instrument_orchestrator',
        output='screen'
    )

    delayed_orchestrator = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=yolo_node,
            on_exit=[orchestrator]
        )
    )

    # ------------------- Instrument Pick and Place -------------------
    picker = Node(
        package='pymoveit2',
        executable='instrument_pick_and_place.py',
        name='instrument_pick_and_place',
        output='screen'
    )

    delayed_picker = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=orchestrator,
            on_exit=[picker]
        )
    )

    # ------------------- Speech Listener -------------------
    speech_listener = Node(
        package='panda_vision',
        executable='speech_listener',
        name='speech_listener',
        output='screen',
        parameters=[{
            'stt_server_port': 5556,
            'stt_server_host': 'localhost',
            'reconnect_interval': 2.0,
            'publish_text': True,
            'publish_instrument': True
        }]
    )

    delayed_speech_listener = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=picker,
            on_exit=[speech_listener]
        )
    )
    
    return LaunchDescription([
        # Arguments
        world_name_arg,
        model_arg,
        randomize_arg,
        random_seed_arg,
        stt_mode_arg,
        llm_mode_arg,
        whisper_model_arg,
        whisper_device_arg,
        gemini_model_arg,

        # Environment
        gazebo_resource_path,

        # External servers
        yolo_server,
        stt_server,

        # Gazebo Core
        robot_state_publisher_node,
        gazebo,
        controller,
        moveit,
        spawn_panda,
        # delayed_spawn_panda,
        # spawn_instruments,
        delayed_spawn_instruments,

        # Bridges
        ros_gz_rgb_bridge,
        ros_gz_depth_bridge,
        ros_gz_bridge,

        # ROS2 Pipeline
        delayed_yolo_node, # ros2 run panda_vision yolo_detector
        llm_controller, # ros2 run panda_vision llm_controller
        delayed_orchestrator, # ros2 run panda_vision instrument_orchestrator
        delayed_picker, # ros2 run pymoveit instrument_pick_and_place
        delayed_speech_listener,  # ros2 run panda_vision speech_listener
    ])