from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    stt_mode = LaunchConfiguration('stt_mode', default='mock')
    stt_port = LaunchConfiguration('stt_port', default='5556')
    whisper_model = LaunchConfiguration('whisper_model', default='base')
    whisper_device = LaunchConfiguration('whisper_device', default='cpu')
    llm_mode = LaunchConfiguration('llm_mode', default='mock')  # NEW
    gemini_model = LaunchConfiguration('gemini_model', default='gemini-3.1-flash-lite')  # NEW
    
    # Get the path to the outside folder (2 levels up from src/panda_vision)
    # This is a bit hacky - alternative is to move stt_server.py into the package
    workspace_dir = '/home/pwnwas/Personal/College/Skripsi/Code/playground_ws'
    
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'stt_mode',
            default_value='mock',
            description='STT mode: mock, whisper, or file'
        ),
        DeclareLaunchArgument(
            'stt_port',
            default_value='5556',
            description='ZeroMQ port for STT server'
        ),
        DeclareLaunchArgument(
            'whisper_model',
            default_value='base',
            description='Whisper model size: tiny, base, small, medium, large'
        ),
        DeclareLaunchArgument(
            'whisper_device',
            default_value='cpu',
            description='Whisper device: cpu or cuda'
        ),
        DeclareLaunchArgument(
            'llm_mode',
            default_value='mock',
            description='Either using mock or Gemini API for LLM backend'
        ),
        DeclareLaunchArgument(
            'gemini_model', 
            default_value='gemini-3.1-flash-lite',
            description='Gemini model'
        ),
        
        # STT Server (external process)
        ExecuteProcess(
            cmd=[
                '/home/pwnwas/miniconda3/envs/colab_mimic/bin/python',
                f'{workspace_dir}/src/panda_vision/../outside/stt_server.py',
                '--mode', stt_mode,
                '--port', stt_port,
                '--model', whisper_model,
                '--device', whisper_device
            ],
            output='screen',
            name='stt_server'
        ),
        
        # ROS Speech Listener Node - FIXED: use 'speech_listener' not 'speech_listener.py'
        Node(
            package='panda_vision',
            executable='speech_listener',  # ← Removed .py
            name='speech_listener',
            output='screen',
            parameters=[{
                'stt_server_port': LaunchConfiguration('stt_port'),
                'stt_server_host': 'localhost',
                'reconnect_interval': 2.0,
                'publish_text': True,
                'publish_instrument': True
            }]
        ),

        # LLM Controller Node - NEW
        Node(
            package='panda_vision',
            executable='llm_controller',
            name='llm_controller',
            output='screen',
            arguments=[
                '--mode', llm_mode,
                '--model', gemini_model
            ],
            parameters=[{
                'use_sim_time': False
            }]
        ),
    ])