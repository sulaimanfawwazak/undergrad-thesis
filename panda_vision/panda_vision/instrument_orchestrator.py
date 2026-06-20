#!/usr/bin/env python3
# panda_vision/panda_vision/instrument_orchestrator.py
"""
Instrument Orchestrator - Bridges LLM commands to pick-and-place execution
Handles pixel-to-robot coordinate transformation using YOLO OBB detections
Maintains orientation information for proper instrument grasping

Subscribes to: 
    - /robot/command (from LLM)
    - /yolo/detection (YOLO OBB detections)
Publishes to:
    - /target_coords (for picker)
    - /sort_command (for batch sorting)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import json
import numpy as np
import tf2_ros
import tf_transformations
from cv_bridge import CvBridge
from charminal import *
import math

# Sorted positions on Table 2 (in robot base frame)
# These are hardcoded positions where instruments will be placed after sorting
SORTED_POSITIONS = {
    "tweezers":       {"x": 0.11, "y": 0.29, "z": 0.740, "yaw": math.pi/2},
    "scalpel":      {"x": 0.30, "y": 0.28, "z": 0.740, "yaw": math.pi/2},
    "scissors":      {"x": 0.06, "y": 0.43, "z": 0.740, "yaw": math.pi},
    "retractor":     {"x": 0.14, "y": 0.42, "z": 0.740, "yaw": math.pi},
    "needle_holder": {"x": 0.23, "y": 0.42, "z": 0.740, "yaw": math.pi},
}

SORTED_POSITIONS_IDEAL = {
    "tweezers":       {"x": 0.11, "y": 0.35, "z": 0.740, "yaw": math.pi/2},
    "scalpel":      {"x": 0.30, "y": 0.30, "z": 0.740, "yaw": math.pi/2},
    "scissors":      {"x": 0.10, "y": 0.43, "z": 0.740, "yaw": math.pi},
    "retractor":     {"x": 0.15, "y": 0.42, "z": 0.740, "yaw": math.pi},
    "needle_holder": {"x": 0.35, "y": 0.42, "z": 0.740, "yaw": math.pi},
}

SORTED_POSITIONS_HOMO = {
    "tweezers":       {"x": 0.117, "y": 0.448, "z": 0.015, "yaw": -math.pi},
    "scalpel":      {"x": 0.311, "y": 0.448, "z": 0.015, "yaw": -math.pi},
    "scissors":      {"x": 0.081, "y": 0.298, "z": 0.015, "yaw": -math.pi/2},
    "retractor":     {"x": 0.203, "y": 0.300, "z": 0.015, "yaw": -math.pi/2},
    "needle_holder": {"x": 0.329, "y": 0.295, "z": 0.015, "yaw": -math.pi/2},
}

# Order for automatic sorting
SORTING_ORDER = ["tweezers", "scalpel", "scissors", "retractor", "needle_holder"]

# Instrument-specific grasp offsets (fine-tuning for different instruments)
# These are in meters, relative to the instrument's center from YOLO OBB
GRASP_OFFSETS = {
    "needle_holder": {"x": 0.0, "y": 0.0, "z": 0.0},  # Center grasp
    "scissors":      {"x": 0.0, "y": 0.0, "z": 0.0},
    "retractor":     {"x": 0.0, "y": 0.0, "z": 0.0},
    "scalpel":       {"x": 0.0, "y": 0.0, "z": 0.0},  # Will adjust after testing
    "tweezers":      {"x": 0.0, "y": 0.0, "z": 0.0}
}

class InstrumentOrchestrator(Node):
    def __init__(self):
        super().__init__('instrument_orchestrator')
        
        # Publishers
        self.picker_pub = self.create_publisher(
            String, '/target_coords', 10
        )
        self.sort_pub = self.create_publisher(
            String, '/sort_command', 10
        )
        self.status_pub = self.create_publisher(
            String, '/orchestrator/status', 10
        )
        
        # Subscribers
        self.llm_sub = self.create_subscription(
            String, '/robot/command', self.llm_callback, 10
        )
        self.yolo_sub = self.create_subscription(
            String, '/yolo/detection', self.yolo_callback, 10
        )
        self.picker_status_sub = self.create_subscription(
            String, '/picker/status', self.sort_callback, 10
        )
        
        # TF2 setup (reused from color_detector_rgbd.py)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Camera intrinsic parameters (from your SDF)
        self.fx = 698.437
        self.fy = 698.437
        self.cx = 640.0
        self.cy = 360.0

        self.H = np.array([
            [ 1.11608817e-03,  2.86646472e-06, -7.13909780e-01],
            [-5.66244750e-07, -1.07019529e-03,  9.17259027e-01],
            [-1.24726329e-06,  1.32649266e-05,  1.00000000e+00]
        ], dtype=np.float32)
        
        # Table height in camera frame (camera is mounted above table)
        # You can adjust this value based on your actual setup
        # Formula: table_height_in_camera = camera_height - table_height
        # Example: if camera is at 1.225m and table at 0.725m, difference is 0.5m
        self.table_height_in_camera = 0.8  # TODO: Calibrate this value
        
        # Robot constants
        self.table_height_z = 0.725  # Table surface height in robot base frame
        self.instrument_height = 0.015  # 15mm collision mesh height
        
        # State
        self.current_instruments = {}  # {instrument_name: {"x": , "y": , "z": , "yaw": , "confidence": }}
        self.instruments_sorted = False
        self.is_sorting = False
        self.sorting_queue = []
        
        self.get_logger().info(f"{COLOR_GREEN}Instrument Orchestrator Started{RESET}")
        self.get_logger().info(f"  Sorted positions defined for {len(SORTED_POSITIONS)} instruments")
        self.get_logger().info(f"  Sorting order: {SORTING_ORDER}")
        self.get_logger().info(f"  Table height in camera frame: {self.table_height_in_camera}m")

        self.get_logger().info(f"{COLOR_CYAN}[ANJAY] Instrument Orchestrator Started with corrected constants:{RESET}")
        self.get_logger().info(f"  Table height: {self.table_height_z}m")
        self.get_logger().info(f"  Camera to table distance: {self.table_height_in_camera}m")
        self.get_logger().info(f"  Instrument height: {self.instrument_height}m")
        self.get_logger().info(f"  Grasp height: {self.table_height_z + (self.instrument_height/2)}m")

    def pixel_to_robot_homography(self, class_name: str, center_x_pix: float, center_y_pix: float, yaw_rad: float = 0.0) -> dict:
        try:

            pixel_pt = np.array([
                center_x_pix,
                center_y_pix,
                1.0
            ])

            world_pt = self.H @ pixel_pt

            world_pt /= world_pt[2]

            x_robot = float(world_pt[0])
            y_robot = float(world_pt[1])

            # grasp_z = self.table_height_z + (self.instrument_height/2)
            # grasp_z = (self.instrument_height/2) # Don't include table_height since panda and instruments are sitting on the same height --> cancel out the height
            grasp_z = self.instrument_height # Don't include table_height since panda and instruments are sitting on the same height --> cancel out the height

            self.get_logger().info(
                f"HOMOGRAPHY: "
                f"[{class_name}] pixel=({center_x_pix:.1f},{center_y_pix:.1f}) "
                f"→ robot=({x_robot:.3f},{y_robot:.3f})"
            )

            return {
                "x": x_robot,
                "y": y_robot,
                "z": grasp_z,
                "yaw": yaw_rad,
                "success": True
            }

        except Exception as e:

            self.get_logger().error(
                f"Homography conversion failed: {e}"
            )

            return {
                "success": False,
                "error": str(e)
            }
    
    def pixel_to_robot(self, center_x_pix: float, center_y_pix: float, yaw_rad: float = 0.0) -> dict:
        """
        Convert pixel coordinates and orientation to robot base frame.
        
        Args:
            center_x_pix: Bounding box center X in pixels
            center_y_pix: Bounding box center Y in pixels
            yaw_rad: Rotation angle from YOLO OBB (radians)
            
        Returns:
            Dictionary with x, y, z, yaw in robot base frame
        """
        try:
            # Step 1: Inverse projection (pixel -> camera frame)
            # Using the pinhole camera model with fixed Z (table height in camera frame)
            Z_cam = self.table_height_in_camera

            # Standard pinhole camera model
            # u = fx * (X_cam / Z_cam) + cx
            # v = fy * (Y_cam / Z_cam) + cy
            # Therefore:
            # X_cam = (u - cx) * Z_cam / fx
            # Y_cam = (v - cy) * Z_cam / fy

            # In camera frame:
            # X_cam: right (positive) / left (negative)
            # Y_cam: down (positive) / up (negative)  
            # Z_cam: forward (positive) from camera
            
            # Note: The sign and orientation might need adjustment based on your camera setup
            # Using the same logic as color_detector_rgbd.py

            u = center_x_pix
            v = center_y_pix

            X_cam = (u - self.cx) * Z_cam / self.fx
            Y_cam = (v - self.cy) * Z_cam / self.fy
            Z_cam = Z_cam

            # In camera frame: X is right/left, Y is down/up, Z is forward
            # For a downward-facing camera, Y_cam should be negative when looking at table
            # But let's keep it as calculated

            # Point in camera frame
            # pt_cam = np.array([X_cam, Y_cam, Z_cam, 1.0])
            pt_cam = np.array([X_cam, -Y_cam, Z_cam, 1.0])  # Negate Y to point down

            
            # Step 2: Transform camera frame -> robot base frame using TF2

            try:
                # Lookup transform from camera to robot base
                now = self.get_clock().now()
                t = self.tf_buffer.lookup_transform(
                    "panda_link0",  # Robot base frame
                    "rgbd_camera_link",  # Camera frame
                    now,
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )

                # Check if transform is valid
                if (t.transform.translation.x == 0.0 and 
                    t.transform.translation.y == 0.0 and 
                    t.transform.translation.z == 0.0):
                    self.get_logger().warn("Received zero transform, using identity")
                    # Fallback: assume camera is mounted at (0.545, 0, 0.8) with rpy (0, 1.57, 0)
                    # This is the known transform from your URDF
                    pt_base = pt_cam.copy()
                    pt_base[0] = pt_cam[0] + 0.545  # X offset
                    pt_base[1] = pt_cam[1] + 0.0    # Y offset
                    pt_base[2] = pt_cam[2] + 0.8    # Z offset
                
                else:
                    self.get_logger().info(f"Cihuy ga zero atau identit")
                    # Convert to transformation matrix
                    trans = np.array([
                        t.transform.translation.x,
                        t.transform.translation.y,
                        t.transform.translation.z
                    ])
                    
                    rot = [
                        t.transform.rotation.x,
                        t.transform.rotation.y,
                        t.transform.rotation.z,
                        t.transform.rotation.w
                    ]
                    
                    # Create 4x4 transformation matrix
                    T_cam_to_base = tf_transformations.quaternion_matrix(rot)
                    T_cam_to_base[:3, 3] = trans
                    
                    # Transform point from camera frame to robot base frame
                    # pt_cam = np.array([X_cam, Y_cam, Z_cam, 1.0])
                    pt_base = T_cam_to_base @ pt_cam
                
                # Step 3: Apply instrument-specific grasp offset (if any)
                # For now, using center point, can be adjusted per instrument
                
                # Debug output
                self.get_logger().info(f"DEBUG: pixel ({u:.1f}, {v:.1f}) -> camera ({X_cam:.3f}, {Y_cam:.3f}, {Z_cam:.3f}) -> robot ({pt_base[0]:.3f}, {pt_base[1]:.3f}, {pt_base[2]:.3f})")

                # Grasp at center of instrument (table height + half instrument height)
                # grasp_z = self.table_height_z + (self.instrument_height / 2) # Grasp at center height
                # grasp_z = (self.instrument_height / 2) # Don't include table_height since panda and instruments are sitting on the same height --> cancel out the height
                grasp_z = self.instrument_height # Don't include table_height since panda and instruments are sitting on the same height --> cancel out the height

                return {
                    "x": pt_base[0],
                    "y": pt_base[1],
                    "z": grasp_z,
                    "yaw": yaw_rad,  # Preserve orientation from YOLO OBB
                    "success": True
                }
                
            except (tf2_ros.LookupException, 
                    tf2_ros.ConnectivityException, 
                    tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn(f"TF lookup failed: {e}")

                # Fallback: Use hardcoded transform from URDF
                # Camera is at (0.545, 0, 0.8) relative to panda_link0 with rpy (0, 1.57, 0)
                # For a point in camera frame, transform to robot base:
                # x_robot = x_cam * cos(90°) + z_cam * sin(90°) + 0.545 = z_cam + 0.545
                # y_robot = y_cam
                # z_robot = -x_cam * sin(90°) + z_cam * cos(90°) + 0.8 = -x_cam + 0.8
                
                x_robot = Z_cam + 0.545  # z_cam becomes x_robot after 90° rotation
                y_robot = -Y_cam         # Negate Y
                z_robot = -X_cam + 0.8   # -x_cam + camera height
                
                pt_base = np.array([x_robot, y_robot, z_robot, 1.0])
                
                self.get_logger().info(
                    f"DEBUG FALLBACK: camera ({X_cam:.3f}, {Y_cam:.3f}, {Z_cam:.3f}) -> "
                    f"robot ({pt_base[0]:.3f}, {pt_base[1]:.3f}, {pt_base[2]:.3f})"
                )
                
                # grasp_z = 0.7325
                grasp_z = 0.015 # Don't take table's height into account
                
                return {
                    "x": pt_base[0],
                    "y": pt_base[1],
                    "z": grasp_z,
                    "yaw": yaw_rad,
                    "success": True
                }
                # return {"success": False, "error": str(e)}
                
        except Exception as e:
            self.get_logger().error(f"Pixel to robot conversion error: {e}")
            return {"success": False, "error": str(e)}
    
    def yolo_callback(self, msg: String):
        """Process YOLO detections and update current instrument positions with orientation"""
        try:
            detections = json.loads(msg.data)

            if detections:
                det = detections[0]
                
                # Calculate the angle according to the longest line of OBB (between x1 and x2, or x1 and x3)
                det_x1, det_y1, det_x2, det_y2, det_x3, det_y3, det_x4, det_y4 = det["xyxyxyxy"]

                y_min = min(det_y1, det_y2, det_y3, det_y4)

                if y_min == det_y1:
                    # Find distance between x1-x2 and x1-x4
                    dist_1 = np.sqrt((det_x1 - det_x2)**2 + (det_y1 - det_y2)**2)
                    dist_2 = np.sqrt((det_x1 - det_x4)**2 + (det_y1 - det_y4)**2)

                    if dist_1 > dist_2:
                        dx = det_x2 - det_x1
                        dy = det_y2 - det_y1
                        angle = np.arctan2(dy, dx)
                    else:
                        dx = det_x4 - det_x1
                        dy = det_y4 - det_y1
                        angle = np.arctan2(dy, dx)

                elif y_min == det_y2:
                    # Find distance between x2-x1 and x2-x3
                    dist_1 = np.sqrt((det_x2 - det_x1)**2 + (det_y2 - det_y1)**2)
                    dist_2 = np.sqrt((det_x2 - det_x3)**2 + (det_y2 - det_y3)**2)

                    if dist_1 > dist_2:
                        dx = det_x1 - det_x2
                        dy = det_y1 - det_y2
                        angle = np.arctan2(dy, dx)
                    else:
                        dx = det_x3 - det_x2
                        dy = det_y3 - det_y2
                        angle = np.arctan2(dy, dx)

                elif y_min == det_y3:
                    # Find distance between x3-x2 and x3-x4
                    dist_1 = np.sqrt((det_x3 - det_x2)**2 + (det_y3 - det_y2)**2)
                    dist_2 = np.sqrt((det_x3 - det_x4)**2 + (det_y3 - det_y4)**2)

                    if dist_1 > dist_2:
                        dx = det_x2 - det_x3
                        dy = det_y2 - det_y3
                        angle = np.arctan2(dy, dx)
                    else:
                        dx = det_x4 - det_x3
                        dy = det_y4 - det_y3
                        angle = np.arctan2(dy, dx)

                elif y_min == det_y4:
                    # Find distance between x4-x1 and x4-x3
                    dist_1 = np.sqrt((det_x4 - det_x1)**2 + (det_y4 - det_y1)**2)
                    dist_2 = np.sqrt((det_x4 - det_x3)**2 + (det_y4 - det_y3)**2)

                    if dist_1 > dist_2:
                        dx = det_x1 - det_x4
                        dy = det_y1 - det_y4
                        angle = np.arctan2(dy, dx)
                    else:
                        dx = det_x3 - det_x4
                        dy = det_y3 - det_y4
                        angle = np.arctan2(dy, dx)

                angle = -1 * angle

                self.get_logger().info(f"DEBUG YOLO: {det['class_name']} - center: ({det['xywhr'][0]:.1f}, {det['xywhr'][1]:.1f}), rot: {angle:.3f} rad")
            
            # Clear previous positions
            self.current_instruments.clear()
            
            # Process each detection
            for det in detections:
                instrument = det['class_name']
                confidence = det['confidence']
                det_x1, det_y1, det_x2, det_y2, det_x3, det_y3, det_x4, det_y4 = det["xyxyxyxy"]
                
                # Get bounding box center from OBB corners
                corners = np.array(det['xyxyxyxy']).reshape(-1, 2)

                y_min = min(det_y1, det_y2, det_y3, det_y4)

                if y_min == det_y1:
                    # Find distance between x1-x2 and x1-x4
                    dist_1 = np.sqrt((det_x1 - det_x2)**2 + (det_y1 - det_y2)**2)
                    dist_2 = np.sqrt((det_x1 - det_x4)**2 + (det_y1 - det_y4)**2)

                    if dist_1 > dist_2:
                        dx = det_x2 - det_x1
                        dy = det_y2 - det_y1
                        angle = np.arctan2(dy, dx)
                    else:
                        dx = det_x4 - det_x1
                        dy = det_y4 - det_y1
                        angle = np.arctan2(dy, dx)

                elif y_min == det_y2:
                    # Find distance between x2-x1 and x2-x3
                    dist_1 = np.sqrt((det_x2 - det_x1)**2 + (det_y2 - det_y1)**2)
                    dist_2 = np.sqrt((det_x2 - det_x3)**2 + (det_y2 - det_y3)**2)

                    if dist_1 > dist_2:
                        dx = det_x1 - det_x2
                        dy = det_y1 - det_y2
                        angle = np.arctan2(dy, dx)
                    else:
                        dx = det_x3 - det_x2
                        dy = det_y3 - det_y2
                        angle = np.arctan2(dy, dx)

                elif y_min == det_y3:
                    # Find distance between x3-x2 and x3-x4
                    dist_1 = np.sqrt((det_x3 - det_x2)**2 + (det_y3 - det_y2)**2)
                    dist_2 = np.sqrt((det_x3 - det_x4)**2 + (det_y3 - det_y4)**2)

                    if dist_1 > dist_2:
                        dx = det_x2 - det_x3
                        dy = det_y2 - det_y3
                        angle = np.arctan2(dy, dx)
                    else:
                        dx = det_x4 - det_x3
                        dy = det_y4 - det_y3
                        angle = np.arctan2(dy, dx)

                elif y_min == det_y4:
                    # Find distance between x4-x1 and x4-x3
                    dist_1 = np.sqrt((det_x4 - det_x1)**2 + (det_y4 - det_y1)**2)
                    dist_2 = np.sqrt((det_x4 - det_x3)**2 + (det_y4 - det_y3)**2)

                    if dist_1 > dist_2:
                        dx = det_x1 - det_x4
                        dy = det_y1 - det_y4
                        angle = np.arctan2(dy, dx)
                    else:
                        dx = det_x3 - det_x4
                        dy = det_y3 - det_y4
                        angle = np.arctan2(dy, dx)

                angle = -1 * angle

                # center_x_pix = np.mean(corners[:, 0])
                # center_y_pix = np.mean(corners[:, 1])
                
                # Note: xywhr format is [x_center, y_center, width, height, rotation]
                rotation_rad = angle
                center_x_pix = det["xywhr"][0]
                center_y_pix = det["xywhr"][1]
                
                # Convert to robot coordinates
                # robot_pose = self.pixel_to_robot(center_x_pix, center_y_pix, rotation_rad)
                robot_pose = self.pixel_to_robot_homography(instrument, center_x_pix, center_y_pix, rotation_rad)
                
                if robot_pose.get('success', False):
                    self.current_instruments[instrument] = {
                        "x": robot_pose["x"],
                        "y": robot_pose["y"],
                        "z": robot_pose["z"],
                        "yaw": robot_pose["yaw"],
                        "confidence": confidence
                    }
                    self.get_logger().info(
                        f"{COLOR_MAGENTA}Found {instrument} at robot coords: ({robot_pose['x']:.3f}, {robot_pose['y']:.3f}) {RESET}"
                        f"{COLOR_MAGENTA}yaw: {math.degrees(robot_pose['yaw']):.1f}° conf: {confidence:.2f}{RESET}"
                    )
                else:
                    self.get_logger().warn(f"Failed to convert position for {instrument}: {robot_pose.get('error')}")
            
            self.get_logger().debug(f"Updated positions for {len(self.current_instruments)} instruments")
            
        except Exception as e:
            self.get_logger().error(f"Error in YOLO callback: {e}")
    
    def publish_status(self, status: str, details: dict = None):
        """Publish orchestrator status"""
        msg = String()
        status_data = {
            "status": status,
            "instruments_sorted": self.instruments_sorted,
            "is_sorting": self.is_sorting,
            "instruments_visible": list(self.current_instruments.keys()),
            "details": details or {}
        }
        msg.data = json.dumps(status_data)
        self.status_pub.publish(msg)
    
    def execute_sort(self):
        """Execute automatic sorting of all instruments"""
        if self.is_sorting:
            self.get_logger().warn("Sorting already in progress")
            return
        
        if self.instruments_sorted:
            self.get_logger().info("Instruments already sorted, skipping sort command")
            return
        
        # Check if all instruments are visible
        missing = [inst for inst in SORTING_ORDER if inst not in self.current_instruments]
        if missing:
            self.get_logger().warn(f"Missing instruments for sorting: {missing}")
            # Continue anyway with available instruments
        
        self.get_logger().info(f"{COLOR_CYAN}Starting automatic sorting of instruments{RESET}")
        self.is_sorting = True
        self.sorting_queue = [inst for inst in SORTING_ORDER if inst in self.current_instruments]
        
        if not self.sorting_queue:
            self.get_logger().error("No instruments detected for sorting")
            self.is_sorting = False
            return
        
        # Publish sort command to start the process
        sort_msg = String()
        sort_msg.data = json.dumps({
            "action": "start_sorting",
            "order": self.sorting_queue
        })
        self.sort_pub.publish(sort_msg)
        self.publish_status("sorting_started", {"order": self.sorting_queue})
        
        # Process first instrument
        self.process_next_sort()
    
    def process_next_sort(self):
        """Process the next instrument in sorting queue"""
        if not self.sorting_queue:
            # Sorting complete
            self.is_sorting = False
            self.instruments_sorted = True
            self.get_logger().info(f"{COLOR_GREEN}Sorting complete! All instruments on Table 2{RESET}")
            self.publish_status("sorting_complete", {"sorted_instruments": SORTING_ORDER})
            return
        
        instrument = self.sorting_queue[0]
        self.get_logger().info(f"{COLOR_CYAN}Sorting {instrument}...{RESET}")
        
        # Check if instrument exists in current positions
        if instrument not in self.current_instruments:
            self.get_logger().warn(f"{instrument} not detected, skipping...")
            self.sorting_queue.pop(0)
            self.process_next_sort()
            return
        
        # Get current pose from YOLO
        current_pose = self.current_instruments[instrument]
        
        # Get target pose from sorted positions
        target_pose = SORTED_POSITIONS_HOMO[instrument]
        
        # Publish to picker for pick-and-place
        # Format: instrument,operation,current_x,current_y,current_z,current_yaw,target_x,target_y,target_z,target_yaw
        # coord_msg = (
        #     f"{instrument},pick_and_place,"
        #     f"{current_pose['x']},{current_pose['y']},{current_pose['z']},{current_pose['yaw']},"
        #     f"{target_pose['x']},{target_pose['y']},{target_pose['z']},{target_pose['yaw']}"
        # )

        coord_msg = (
            f"{instrument},sort,"
            f"{current_pose['x']},{current_pose['y']},{current_pose['z']},{current_pose['yaw']}"
        )

        self.get_logger().info(f"{COLOR_CYAN}[WOI] coord_msg: {coord_msg}{RESET}")
        
        self.picker_pub.publish(String(data=coord_msg))
        self.publish_status("sorting_in_progress", {
            "current_instrument": instrument,
            "remaining": len(self.sorting_queue) - 1
        })
    
    def execute_pick(self, instrument: str, source_table: str):
        """Execute single pick operation"""
        if source_table == "table2" and not self.instruments_sorted:
            self.get_logger().warn(f"Cannot pick from Table 2 - instruments not sorted yet")
            return False
        
        # Get position
        if source_table == "table2":
            # Pick from sorted position
            if instrument not in SORTED_POSITIONS:
                self.get_logger().error(f"{instrument} not in sorted positions")
                return False
            # pos = SORTED_POSITIONS_HOMO[instrument]
            pos = self.current_instruments[instrument] # Using YOLO detection despite sorted
            source_desc = "sorted table"
            operation = "pick_only_sorted"
        else:
            # Pick from current YOLO position
            if instrument not in self.current_instruments:
                self.get_logger().warn(f"{instrument} not detected in current view")
                return False
            pos = self.current_instruments[instrument]
            source_desc = "current table"
            operation = "pick_only"

        # Add debug logging
        self.get_logger().info(f"{COLOR_CYAN}DEBUG: {instrument} position - x: {pos['x']:.3f}, y: {pos['y']:.3f}, z: {pos['z']:.3f}, yaw: {pos.get('yaw', 0.0):.3f}{RESET}")

        # Publish to picker
        # Format: instrument,operation,x,y,z,yaw
        coord_msg = f"{instrument},{operation},{pos['x']},{pos['y']},{pos['z']},{pos.get('yaw', 0.0)}"
        self.get_logger().info(f"{COLOR_MAGENTA}[WOI] coord_msg: {coord_msg}{RESET}")
        self.picker_pub.publish(String(data=coord_msg))
        
        self.get_logger().info(f"{COLOR_GREEN}Picking {instrument} from {source_desc}{RESET}")
        return True
    
    def llm_callback(self, msg: String):
        """Process LLM commands"""
        try:
            command = json.loads(msg.data)
            operation = command.get('operation', 'none')
            instrument = command.get('instrument', None)
            confidence = command.get('confidence', 0.0)
            
            self.get_logger().info(f"{COLOR_CYAN}Received LLM command: {operation} {instrument or ''} (conf: {confidence:.2f}){RESET}")
            
            if operation == 'pick':
                if instrument:
                    self.execute_pick(instrument, "table1")
                else:
                    self.get_logger().warn("Pick command missing instrument")
                    
            elif operation == 'pick_sorted':
                if instrument:
                    self.execute_pick(instrument, "table2")
                else:
                    self.get_logger().warn("Pick_sorted command missing instrument")
                    
            elif operation == 'sort':
                self.execute_sort()
                
            elif operation == 'none':
                self.get_logger().info(f"No action needed: {command.get('reasoning', '')}")
                
            else:
                self.get_logger().warn(f"Unknown operation: {operation}")
                
        except Exception as e:
            self.get_logger().error(f"Error processing LLM command: {e}")
    
    def sort_callback(self, msg: String):
        """Callback for when sorting of one instrument is complete"""
        try:
            data = json.loads(msg.data)
            if data.get('status') == 'complete':
                # Remove the completed instrument from queue
                if self.sorting_queue:
                    completed = self.sorting_queue.pop(0)
                    self.get_logger().info(f"Completed sorting for {completed}")
                    
                    # Process next instrument
                    self.process_next_sort()
                    
        except Exception as e:
            self.get_logger().error(f"Error in sort callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = InstrumentOrchestrator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()