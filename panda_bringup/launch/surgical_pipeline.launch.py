# panda_bringup/surgical_pipeline.launch.py
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # Launch arguments
    stt_mode = LaunchConfiguration('stt_mode', default='mock')
    llm_mode = LaunchConfiguration('llm_mode', default='mock')
    whisper_model = LaunchConfiguration('whisper_model', default='base')
    whisper_device = LaunchConfiguration('whisper_device', default='cpu')
    gemini_model = LaunchConfiguration('gemini_model', default='gemini-3.1-flash-lite')
    
    workspace_dir = '/home/pwnwas/Personal/College/Skripsi/Code/playground_ws'
    
    # ------------------- Gazebo -------------------
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("panda_description"),
                "launch",
                "panda_with_instruments.launch.py"
            )
        ),
        launch_arguments={"randomize_instruments": "true"}.items()
    )
    
    # ------------------- Controllers -------------------
    controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("panda_controller"),
                "launch",
                # "panda_controller.launch.py"
                "slider_controller.launch.py"
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
    
    # ------------------- YOLO Detector -------------------
    yolo_node = Node(
        package="panda_vision",
        executable="yolo_detector",
        name="yolo_detector",
        output="screen"
    )
    
    # ------------------- STT Server (external) -------------------
    stt_server = ExecuteProcess(
        cmd=[
            '/home/pwnwas/miniconda3/envs/colab_mimic/bin/python',
            f'{workspace_dir}/src/panda_vision/../outside/stt_server.py',
            '--mode', stt_mode,
            '--model', whisper_model,
            '--device', whisper_device
        ],
        output='screen',
        name='stt_server'
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
    
    # ------------------- LLM Controller -------------------
    llm_controller = Node(
        package='panda_vision',
        executable='llm_controller',
        name='llm_controller',
        output='screen',
        arguments=[
            '--mode', llm_mode,
            '--model', gemini_model
        ]
    )
    
    # ------------------- Instrument Orchestrator -------------------
    orchestrator = Node(
        package='panda_vision',
        executable='instrument_orchestrator',
        name='instrument_orchestrator',
        output='screen'
    )
    
    # ------------------- Instrument Pick and Place -------------------
    picker = Node(
        package='pymoveit2',
        executable='instrument_pick_and_place.py',
        name='instrument_pick_and_place',
        output='screen'
    )
    
    return LaunchDescription([
        DeclareLaunchArgument('stt_mode', default_value='mock'),
        DeclareLaunchArgument('llm_mode', default_value='mock'),
        DeclareLaunchArgument('whisper_model', default_value='base'),
        DeclareLaunchArgument('whisper_device', default_value='cpu'),
        DeclareLaunchArgument('gemini_model', default_value='gemini-3.1-flash-lite'),
        
        gazebo,
        controller,
        moveit,
        yolo_node,
        stt_server,
        speech_listener,
        llm_controller,
        orchestrator,
        picker,
    ])