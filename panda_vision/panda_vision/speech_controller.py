#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import zmq
import json
import threading
import time
import sys
import tty
import termios
import select

class SpeechController(Node):
    def __init__(self):
        super().__init__('speech_controller')
        
        # Publisher for transcribed commands
        self.command_pub = self.create_publisher(String, '/speech/command', 10)
        
        # Publisher for recording status
        self.status_pub = self.create_publisher(String, '/speech/status', 10)
        
        # ZeroMQ client for YOLO server
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5555")
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        
        # State
        self.is_recording = False
        self.last_command = ""
        self.recording_lock = threading.Lock()
        
        self.get_logger().info("Speech Controller Started")
        self.get_logger().info("Press and hold 'R' to record, release to transcribe")
        self.get_logger().info("Press 'T' to test with audio file")
        self.get_logger().info("Press 'Q' to quit")
        
        # Start keyboard listener thread
        self.keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.keyboard_thread.start()
    
    def _send_zmq_request(self, request_type, **kwargs):
        """Send request to YOLO server and return response"""
        request = {'type': request_type}
        request.update(kwargs)
        
        try:
            self.socket.send(json.dumps(request).encode())
            response = json.loads(self.socket.recv().decode())
            return response
        except Exception as e:
            self.get_logger().error(f"ZMQ request failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def start_recording(self):
        """Start recording (key press)"""
        with self.recording_lock:
            if self.is_recording:
                return
        
        self.get_logger().info("Recording started...")
        response = self._send_zmq_request('start_recording')
        
        if response.get('success'):
            with self.recording_lock:
                self.is_recording = True
            # Publish status
            status_msg = String()
            status_msg.data = "recording"
            self.status_pub.publish(status_msg)
        else:
            self.get_logger().error(f"Failed to start recording: {response.get('error')}")
    
    def stop_and_transcribe(self):
        """Stop recording and get transcription (key release)"""
        with self.recording_lock:
            if not self.is_recording:
                return
            self.is_recording = False
        
        self.get_logger().info("Stopping recording and transcribing...")
        response = self._send_zmq_request('stop_and_transcribe')
        
        if response.get('success'):
            command = response.get('caption', '')
            self.last_command = command
            self.get_logger().info(f"Transcribed: '{command}'")
            
            # Publish command for LLM processing
            command_msg = String()
            command_msg.data = command
            self.command_pub.publish(command_msg)
            
            # Publish status
            status_msg = String()
            status_msg.data = "idle"
            self.status_pub.publish(status_msg)
        else:
            error_msg = response.get('error', 'Unknown error')
            self.get_logger().warn(f"Transcription failed: {error_msg}")
            status_msg = String()
            status_msg.data = "idle"
            self.status_pub.publish(status_msg)
    
    def test_audio_file(self, file_path):
        """Test with pre-recorded audio file"""
        self.get_logger().info(f"Testing with audio file: {file_path}")
        
        # Read audio file
        import base64
        try:
            with open(file_path, 'rb') as f:
                audio_data = base64.b64encode(f.read()).decode()
        except FileNotFoundError:
            self.get_logger().error(f"Test file not found: {file_path}")
            return
        
        response = self._send_zmq_request('transcribe_file', audio_file=audio_data)
        
        if response.get('success'):
            command = response.get('caption', '')
            self.get_logger().info(f"Test transcription: '{command}'")
            
            # Publish test command
            command_msg = String()
            command_msg.data = f"[TEST] {command}"
            self.command_pub.publish(command_msg)
        else:
            self.get_logger().error(f"Test failed: {response.get('error')}")
    
    def _keyboard_listener(self):
        """Listen for keyboard input in a non-blocking way"""
        # Check if we're in a terminal
        if not sys.stdin.isatty():
            self.get_logger().error("Not running in a terminal! Keyboard input won't work.")
            self.get_logger().info("Please run this node in a real terminal, not through ros2 launch")
            return
        
        # Save terminal settings
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            tty.setraw(fd)
            r_pressed = False
            
            while rclpy.ok():
                # Check if key is pressed
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    
                    if key.lower() == 'r' and not r_pressed:
                        # R key pressed - start recording
                        r_pressed = True
                        self.start_recording()
                    
                    elif key.lower() == 't':
                        # T key for test
                        test_file = "/home/pwnwas/test_commands/scalpel_command.wav"
                        self.test_audio_file(test_file)
                    
                    elif key.lower() == 'q':
                        self.get_logger().info("Quitting...")
                        rclpy.shutdown()
                        break
                
                # Check for key release (when no key is pressed but r_pressed is True)
                elif r_pressed and not select.select([sys.stdin], [], [], 0.01)[0]:
                    r_pressed = False
                    self.stop_and_transcribe()
                
                time.sleep(0.01)
        
        except Exception as e:
            self.get_logger().error(f"Keyboard listener error: {e}")
        finally:
            # Restore terminal settings
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def main(args=None):
    rclpy.init(args=args)
    node = SpeechController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()