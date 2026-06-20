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

class YOLODetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        
        # Frame rate control
        self.frame_counter = 0
        self.inference_interval = 30  # Process every 30 frames
        
        # Subscriber
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10
        )
        
        # Publisher
        self.detection_pub = self.create_publisher(String, "/yolo/detection", 10)
        
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
        
        # OpenCV bridge
        self.bridge = CvBridge()
        
        # Store latest detections
        self.latest_detections = []
        
        self.get_logger().info("YOLO Detector Node Started")
    
    def _process_requests(self):
        """Async worker to handle YOLO requests"""
        while True:
            try:
                frame = self.request_queue.get(timeout=1)
                
                # Encode image to JPEG
                _, encoded = cv2.imencode('.jpg', frame)
                image_data = base64.b64encode(encoded).decode()
                
                # Send request
                request = json.dumps({
                    'type': 'inference',
                    'image': image_data
                })
                
                self.socket.send(request.encode())
                
                # Wait for response
                response = self.socket.recv()
                result = json.loads(response.decode())
                
                if result['success']:
                    self.latest_detections = result['detections']
                    print(f"{COLOR_CYAN}[pwn] result['detections']: {result['detections']}{RESET}")
                    
                    # Publish results
                    detection_msg = String()
                    detection_msg.data = json.dumps(result['detections'])
                    self.detection_pub.publish(detection_msg)
                    
                    self.get_logger().info(f"Detected {len(result['detections'])} objects")
                
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"YOLO request failed: {e}")
    
    def draw_obb(self, frame, detections):
        """Draw oriented bounding boxes"""
        class_colors = {
            "needle_holder": (0, 0, 255),
            "scalpel": (0, 255, 0),
            "scissors": (255, 255, 255),
            "tweezers": (215, 65, 245),
            "retractor": (240, 245, 60),
        }
        
        for detection in detections:
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            
            corners = detection["xyxyxyxy"]
            corners_array = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))

            color = class_colors.get(class_name, (0, 0, 0))
            
            # Draw box
            cv2.polylines(frame, [corners_array], isClosed=True, color=color, thickness=2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            cv2.rectangle(
                frame,
                (corners_array[0][0][0], corners_array[0][0][1] - text_height - 5),
                (corners_array[0][0][0] + text_width, corners_array[0][0][1]),
                color,
                -1
            )
            
            cv2.putText(
                frame,
                label,
                (corners_array[0][0][0], corners_array[0][0][1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
    
    def image_callback(self, msg):
        self.frame_counter += 1
        
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return
        
        # Queue inference request every N frames (non-blocking)
        if self.frame_counter % self.inference_interval == 0:
            try:
                # Non-blocking queue (won't block if full)
                self.request_queue.put_nowait(frame.copy())
            except queue.Full:
                pass  # Skip frame if queue is full
        
        # Draw and display
        if self.latest_detections:
            self.draw_obb(frame, self.latest_detections)
        
        try:
            cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLO Detection", 1280, 720)
            cv2.imshow("YOLO Detection", frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f"OpenCV display error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = YOLODetector()
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