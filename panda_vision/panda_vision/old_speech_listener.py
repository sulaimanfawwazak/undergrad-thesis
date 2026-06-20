#!/usr/bin/env python3
"""
ROS2 Node for receiving speech recognition results
Subscribes to STT server via ZeroMQ and publishes to /speech/text
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import zmq
import threading
import time
from charminal import *
import random

class SpeechListener(Node):
    def __init__(self):
        super().__init__('speech_listener')
        
        # Parameters
        self.declare_parameter('stt_server_port', 5556)
        self.declare_parameter('stt_server_host', 'localhost')
        self.declare_parameter('reconnect_interval', 2.0)  # seconds
        self.declare_parameter('publish_text', True)
        self.declare_parameter('publish_instrument', True)
        
        self.stt_port = self.get_parameter('stt_server_port').value
        self.stt_host = self.get_parameter('stt_server_host').value
        self.reconnect_interval = self.get_parameter('reconnect_interval').value
        self.publish_text = self.get_parameter('publish_text').value
        self.publish_instrument = self.get_parameter('publish_instrument').value
        
        # Publishers
        self.text_pub = self.create_publisher(String, '/speech/text', 10)
        self.instrument_pub = self.create_publisher(String, '/speech/instrument', 10)
        self.raw_pub = self.create_publisher(String, '/speech/raw', 10)
        
        # ZeroMQ setup
        self.zmq_context = zmq.Context()
        self.socket = None
        self.connected = False
        self.running = True
        
        # Instrument mapping (for quick lookup)
        self.instruments = ["needle_holder", "scalpel", "scissors", "tweezers", "retractor"]
        
        # Start connection thread
        self.connection_thread = threading.Thread(target=self._maintain_connection, daemon=True)
        self.connection_thread.start()
        
        # Start listener thread
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()
        
        self.get_logger().info(f"Speech Listener Node Started")
        self.get_logger().info(f"  STT Server: {self.stt_host}:{self.stt_port}")
        
    def _maintain_connection(self):
        """Maintain ZeroMQ connection with reconnect logic"""
        while self.running:
            if not self.connected:
                try:
                    self.socket = self.zmq_context.socket(zmq.REQ)
                    self.socket.connect(f"tcp://{self.stt_host}:{self.stt_port}")
                    self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
                    self.socket.setsockopt(zmq.SNDTIMEO, 1000)
                    
                    # Test connection with ping
                    self.socket.send(json.dumps({"type": "ping"}).encode())
                    response = self.socket.recv()
                    result = json.loads(response.decode())
                    
                    if result.get('success'):
                        self.connected = True
                        self.get_logger().info(f"Connected to STT server (mode: {result.get('mode', 'unknown')})")
                    else:
                        self.get_logger().warn(f"STT server ping failed: {result.get('error')}")
                        self.socket.close()
                        self.socket = None
                        
                except zmq.ZMQError as e:
                    self.get_logger().warn(f"Failed to connect to STT server: {e}")
                    if self.socket:
                        self.socket.close()
                        self.socket = None
                except Exception as e:
                    self.get_logger().error(f"Connection error: {e}")
                    
            time.sleep(self.reconnect_interval)
    
    def extract_instrument(self, text: str) -> str:
        """Extract instrument name from transcribed text"""
        text_lower = text.lower()
        
        # Check each instrument
        for instrument in self.instruments:
            # Direct match
            if instrument.replace('_', ' ') in text_lower:
                return instrument
            
            # Alternative names
            if instrument == "needle_holder" and ("needle holder" in text_lower or "needle driver" in text_lower):
                return instrument
            if instrument == "scalpel" and ("scalpel" in text_lower or "blade" in text_lower):
                return instrument
            if instrument == "scissors" and ("scissors" in text_lower or "scissor" in text_lower or "snip" in text_lower):
                return instrument
            if instrument == "tweezers" and ("tweezers" in text_lower or "forceps" in text_lower):
                return instrument
            if instrument == "retractor" and ("retractor" in text_lower or "retract" in text_lower):
                return instrument
                
        return "unknown"
    
    def _listen_loop(self):
        """Main listening loop"""
        while self.running:
            if not self.connected or not self.socket:
                time.sleep(0.5)
                continue
                
            try:
                # Request transcription
                request = json.dumps({"type": "transcribe"})
                self.socket.send(request.encode())
                
                # Wait for response
                response = self.socket.recv()
                result = json.loads(response.decode())
                
                if result.get('success'):
                    text = result.get('text', '')
                    confidence = result.get('confidence', 0.0)
                    instrument = result.get('instrument', '')
                    
                    # Extract instrument if not provided
                    if not instrument or instrument == "unknown":
                        instrument = self.extract_instrument(text)
                    
                    # Log the result
                    mock_tag = "[MOCK]" if result.get('mock') else "[REAL]"
                    self.get_logger().info(f"{mock_tag} '{text}' (conf: {confidence:.2f}) -> {instrument}")
                    
                    # Publish to ROS topics
                    if self.publish_text and text:
                        msg = String()
                        msg.data = text
                        self.text_pub.publish(msg)
                    
                    if self.publish_instrument and instrument != "unknown":
                        msg = String()
                        msg.data = instrument
                        self.instrument_pub.publish(msg)
                    
                    # Raw message with metadata
                    raw_msg = String()
                    raw_msg.data = json.dumps({
                        "text": text,
                        "confidence": confidence,
                        "instrument": instrument,
                        "mock": result.get('mock', False),
                        "timestamp": self.get_clock().now().to_msg().sec
                    })
                    self.raw_pub.publish(raw_msg)

                    if result.get('mock'):
                        random_interval = random.uniform(60.0, 65.0)
                        print(f"{COLOR_MAGENTA}[pwn] Interval: {random_interval}{RESET}")
                        time.sleep(random_interval)
                    
                elif result.get('error'):
                    # No speech detected, just continue
                    pass
                                
            except zmq.Again:
                # Timeout, no response
                pass
            except zmq.ZMQError as e:
                if self.connected:
                    self.get_logger().warn(f"ZMQ error: {e}, reconnecting...")
                    self.connected = False
                    if self.socket:
                        self.socket.close()
                        self.socket = None
            except Exception as e:
                self.get_logger().error(f"Error in listen loop: {e}")
                time.sleep(0.5)
    
    def destroy_node(self):
        self.running = False
        if self.socket:
            self.socket.close()
        self.zmq_context.term()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SpeechListener()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()