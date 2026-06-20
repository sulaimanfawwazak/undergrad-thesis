#!/usr/bin/env python3
# panda_vision/panda_vision/yolo_detector.py
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import tf2_ros
import tf_transformations
import subprocess
import json

class YOLODetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        self.frame_counter = 0
        self.inference_time_interval = 0.5
        self.fps = 30
        self.frame_inference_interval = self.inference_time_interval / (1/self.fps)
        # self.frame_inference_interval = 30 # Image streams at 30 fps --> 30 frames := 1 second

        # Subscriber
        self.image_sub = self.create_subscription(
            Image,
            # '/camera/image_raw',
            '/camera/image',
            self.image_callback,
            10
        )

        # Publisher
        self.detection_pub = self.create_publisher(String, "/yolo/detection", 10)

        self.conda_python = "/home/pwnwas/miniconda3/envs/colab_mimic/bin/python"
        self.yolo_script = "/home/pwnwas/Personal/College/Skripsi/Code/playground_ws/src/panda_vision/panda_vision/yolo_inference.py"

        # OpenCV bridge
        self.bridge = CvBridge()

        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Camera intrinsic parameters (from your SDF)
        self.fx = 585.0
        self.fy = 588.0
        self.cx = 320.0
        self.cy = 160.0

        # Store latest detections for display
        self.latest_detections = []

        self.get_logger().info("YOLO Detector Node Started")

    def draw_obb(self, frame, detections):
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
            corners_array = np.array(corners, dtype=np.int32).reshape((-1, 1, 2)) # Reshape to points array
            
            color = class_colors.get(class_name, (0, 0, 0))

            # Draw the oriented bounding box
            cv2.polylines(frame, [corners_array], isClosed=True, color=color, thickness=2)

            label = f"{class_name}: {confidence:.2f}"

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Draw label background (fixed with color and filled)
            cv2.rectangle(
                frame,
                (corners_array[0][0][0], corners_array[0][0][1] - text_height - 5),
                (corners_array[0][0][0] + text_width, corners_array[0][0][1]),
                color,
                -1  # Filled rectangle
            )

            # Draw text
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
            # Convert ROS Image -> OpenCV BGR
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return
        
        # Try to get the frame every 30 frames or 1 second
        if self.frame_counter % self.frame_inference_interval == 0:
          # Save the current frame
          tmp_img_path = "/tmp/current_frame.jpg"
          cv2.imwrite(tmp_img_path, frame)

          try:
            # IMPORTANT! POSSIBLE BOTTOLENECK CAUSE
            result = subprocess.run(
                [self.conda_python, self.yolo_script, "--image", tmp_img_path],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                try:
                    if not result.stdout.strip():
                        self.get_logger().error("YOLO inference returned empty stdout")
                        self.get_logger().error(f"stderr: {result.stderr}")
                        return
                    
                    detections = json.loads(result.stdout)
                    self.latest_detections = detections

                    # Publish detections
                    detection_msg = String()
                    detection_msg.data = result.stdout
                    self.detection_pub.publish(detection_msg)
                    
                    self.get_logger().info(f"Detected {len(detections)} objects")
                    self.get_logger().info(f"Detections: {result.stdout}")
                
                except json.JSONDecodeError as e:
                    self.get_logger().error(f"Failed to parse JSON: {e}")
                    self.get_logger().error(f"Raw output: {result.stdout}")
            
            else:
                self.get_logger().error(f"Error: {result.stderr}")
            
          except subprocess.TimeoutExpired:
            self.get_logger().error("YOLO inference timed out")
          except Exception as e:
            self.get_logger().error(f"Subprocess error: {e}")

        if self.latest_detections:
            # frame = self.draw_obb(frame, self.latest_detections)
            self.draw_obb(frame, self.latest_detections)

        # Show image in window
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