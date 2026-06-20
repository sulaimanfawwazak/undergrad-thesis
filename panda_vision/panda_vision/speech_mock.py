#!/usr/bin/env python3
# panda_vision/panda_vision/yolo_detector.py
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import zmq
import base64
import json
from threading import Thread
import queue
import tf2_ros
from charminal import *

class SpeechMock(Node):
    def __init__(self):
        super().__init__('speech_mock')
        
        # Publisher
        self.caption_pub = self.create_publisher(String, "/speech/caption", 10)
        
        # ZeroMQ client
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5555")
        self.socket.setsockopt(zmq.RCVTIMEO, 3000)  # 3 second timeout
        
        # Async request queue
        # self.request_queue = queue.Queue()
        self.request_queue = queue.Queue(maxsize=1)
        self.response_queue = queue.Queue()
        
        # Start async worker thread
        self.worker_thread = Thread(target=self._process_requests, daemon=True)
        self.worker_thread.start()
        
        # Store latest detections
        self.latest_caption = []
        
        self.get_logger().info("Speech Mock Node Started")
    
    def _process_requests(self):
        """Async worker to handle YOLO requests"""
        while True:
            try:
                speech = self.request_queue.get(timeout=1)
                
                # Encode speech to JPEG
                speech_data = base64.b64encode(speech).decode()
                
                # Send request
                request = json.dumps({
                    'type': 'speech',
                    'image': speech_data
                })
                
                self.socket.send(request.encode())
                
                # Wait for response
                response = self.socket.recv()
                result = json.loads(response.decode())
                
                if result['success']:
                    self.latest_caption = result['caption']
                    print(f"{COLOR_CYAN}[pwn] result['caption']: {result['caption']}{RESET}")
                    
                    # Publish results
                    caption_msg = String()
                    caption_msg.data = json.dumps(result['caption'])
                    self.caption_pub.publish(caption_msg)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Speech recognition failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SpeechMock()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()