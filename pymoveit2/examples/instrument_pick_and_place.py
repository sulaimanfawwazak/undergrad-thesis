#!/usr/bin/env python3
# pymoveit2/examples/instrument_pick_and_place.py
"""
Instrument pick and place node for surgical instruments.
Handles both single picks and batch sorting operations with orientation support.
Uses YOLO OBB orientation to approach instruments at the correct angle.

Usage (called by orchestrator, not directly):
    ros2 run pymoveit2 instrument_pick_and_place.py
"""

from threading import Thread
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String
import json
import math
import numpy as np
from geometry_msgs.msg import Quaternion
import tf_transformations
from charminal import *
import time
import datetime

from pymoveit2 import MoveIt2, GripperInterface
from pymoveit2.robots import panda

# Instrument height (meters)
INSTRUMENT_HEIGHT = 0.015  # 15mm

# Approach distance above instrument (meters)
APPROACH_HEIGHT_OFFSET = 0.30  # 10cm above to avoid collision

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

DROP_ZONE = {"x": 0, "y": -0.335, "z": 0 + APPROACH_HEIGHT_OFFSET, "yaw": 0.0}

class InstrumentPickAndPlace(Node):
    def __init__(self):
        super().__init__("instrument_pick_and_place")
        
        # Parameters
        self.declare_parameter("approach_offset", 0.31)
        self.approach_offset = float(self.get_parameter("approach_offset").value)
        
        self.declare_parameter("lift_height", 0.15)
        self.lift_height = float(self.get_parameter("lift_height").value)
        
        # State
        self.current_task = None
        self.current_instrument = None
        self.is_busy = False # Track if currently executing a pick operation
        
        self.callback_group = ReentrantCallbackGroup()
        
        # Arm MoveIt2 interface
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=panda.joint_names(),
            base_link_name=panda.base_link_name(),
            end_effector_name=panda.end_effector_name(),
            group_name=panda.MOVE_GROUP_ARM,
            callback_group=self.callback_group,
        )
        
        # Set lower velocity & acceleration for smoother motion
        self.moveit2.max_velocity = 0.8
        self.moveit2.max_acceleration = 0.8
        
        # Gripper interface
        self.gripper = GripperInterface(
            node=self,
            gripper_joint_names=panda.gripper_joint_names(),
            open_gripper_joint_positions=panda.OPEN_GRIPPER_JOINT_POSITIONS,
            closed_gripper_joint_positions=panda.CLOSED_GRIPPER_JOINT_POSITIONS,
            gripper_group_name=panda.MOVE_GROUP_GRIPPER,
            callback_group=self.callback_group,
            gripper_command_action_name="gripper_action_controller/gripper_cmd",
        )
        
        # Subscriber for target commands
        self.target_sub = self.create_subscription(
            String, '/target_coords', self.target_callback, 10
        )
        
        # Publisher for status
        self.status_pub = self.create_publisher(
            String, '/picker/status', 10
        )
        
        # Predefined joint positions (in radians)
        self.start_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.radians(-125.0)]
        self.home_joints = [0.0, 0.0, 0.0, math.radians(-90.0), 0.0, math.radians(92.0), math.radians(50.0)]
        self.drop_joints = [math.radians(-155.0), math.radians(30.0), math.radians(-20.0),
                            math.radians(-124.0), math.radians(44.0), math.radians(163.0), math.radians(7.0)]
        
        # Move to start joint configuration
        self.moveit2.move_to_configuration(self.start_joints)
        self.moveit2.wait_until_executed()
        
        self.get_logger().info("Instrument Pick and Place Node Ready")
        self.get_logger().info(f"  Approach offset: {self.approach_offset}m")
        self.get_logger().info(f"  Lift height: {self.lift_height}m")
        self.get_logger().info(f"  Instrument height: {INSTRUMENT_HEIGHT}m")
    
    def euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> Quaternion:
        """Convert Euler angles to quaternion"""
        quat = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
        msg = Quaternion()
        msg.x = quat[0]
        msg.y = quat[1]
        msg.z = quat[2]
        msg.w = quat[3]
        return msg
    
    def get_grasp_orientation(self, yaw_rad: float, adjust: bool=True) -> Quaternion:
        """
        Get end-effector orientation for grasping.
        For surgical instruments, we want the gripper to align with the instrument's orientation.
        For Panda robot, end-effector should point straight DOWN.
        Yaw rotation is applied around the vertical axis.
        
        Args:
            yaw_rad: Instrument orientation angle in radians (from YOLO OBB)
            
        Returns:
            Quaternion for MoveIt2 pose goal
        """


        if adjust:
            yaw_deg = math.degrees(yaw_rad)
            # yaw_deg = -1 * yaw_deg
            yaw_deg += 90

            yaw_rad = math.radians(yaw_deg)
        
        # For Panda robot, we want the gripper to approach from above
        # The orientation should have:
        # - Roll: 0 (gripper facing forward)
        # - Pitch: pi/2 (90 degrees) to point gripper downward? 
        # - Yaw: instrument orientation (align gripper with instrument)

        # Common Panda orientation for picking: gripper facing down
        # This means the end-effector's Z axis points toward the table
        # Typically: roll=pi, pitch=0, yaw=yaw_rad (or variations)
        
        # You may need to adjust this based on your gripper's neutral orientation
        # For most setups, approaching from above with aligned yaw works well
        # msg = Quaternion()
        # msg.x = 0.0
        # msg.y = 1.0
        # msg.z = 0.0
        # msg.w = 0.0
        # return [0.0, 1.0, 0.0, 0.0] # Euler: (3.141592653589793, -0.0, 3.141592653589793)
        # return msg
        return self.euler_to_quaternion(math.pi, 0.0, yaw_rad) # Roll 180° to point gripper down

    def publish_status(self, status: str, instrument: str = None, message: str = ""):
        """Publish picker status"""
        status_msg = String()
        data = {
            "status": status,
            "instrument": instrument,
            "message": message
        }
        status_msg.data = json.dumps(data)
        self.status_pub.publish(status_msg)
    
    def pick_and_sort(self, pick_pose: dict, instrument: str):
        """
        Pick instrument from pick_pose and move to its sorted position.
        
        Args:
            pick_pose: {'x', 'y', 'z', 'yaw'} - Position and orientation to pick from (in Panda frame)
            instrument: Name of the instrument
        """
        # Get sorted position for this instrument
        if instrument not in SORTED_POSITIONS_HOMO:
            self.get_logger().error(f"No sorted position defined for {instrument}")
            return False
        
        place_pose = SORTED_POSITIONS_HOMO[instrument]
        
        self.get_logger().info(
            f"{COLOR_GREEN}Sorting {instrument} to sorted position "
            f"({place_pose['x']:.3f}, {place_pose['y']:.3f}, {place_pose['z']}, {math.degrees(place_pose['yaw'])}° / {place_pose['yaw']} rad){RESET}"
        )
        
        # Use pick_and_place_to_target with the sorted position
        return self.pick_and_place_to_target(pick_pose, place_pose, instrument)

    def pick_and_place_to_target(self, pick_pose: dict, place_pose: dict, instrument: str):
        """
        Pick instrument from pick_pose and place at place_pose.
        
        Args:
            pick_pose: {'x', 'y', 'z', 'yaw'} - Position and orientation to pick from
            place_pose: {'x', 'y', 'z', 'yaw'} - Position and orientation to place at
            instrument: Name of the instrument
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get quaternion orientations
            pick_quat = self.get_grasp_orientation(pick_pose['yaw'])
            place_quat = self.get_grasp_orientation(place_pose['yaw'], adjust=True)
            
            # Calculate positions
            pick_approach = [
                pick_pose['y'],
                -pick_pose['x'],
                pick_pose['z'] + APPROACH_HEIGHT_OFFSET
            ]
            pick_at_height = [
                pick_pose['y'],
                -pick_pose['x'],
                pick_pose['z'] + 0.11
            ]
            lift_position = [
                pick_pose['y'],
                -pick_pose['x'],
                # pick_pose['z'] + self.lift_height
                pick_pose['z'] + APPROACH_HEIGHT_OFFSET
            ]
            # place_approach = [
            #     place_pose['y'],
            #     -place_pose['x'],
            #     place_pose['z'] + APPROACH_HEIGHT_OFFSET
            # ]
            # place_at_height = [
            #     place_pose['y'],
            #     -place_pose['x'],
            #     place_pose['z'] + 0.2
            # ]
            place_approach = [
                place_pose['y'],
                -place_pose['x'],
                place_pose['z'] + APPROACH_HEIGHT_OFFSET
            ]
            place_at_height = [
                place_pose['y'],
                -place_pose['x'],
                place_pose['z'] + 0.15
            ]
            
            # --- Pick sequence ---
            
            # # 1. Move to start
            # self.get_logger().info(f"{COLOR_MAGENTA}Returning to start position...{RESET}")
            # self.moveit2.move_to_configuration(self.start_joints)
            # self.moveit2.wait_until_executed()

            # # 2. Move to home
            # self.get_logger().info(f"{COLOR_MAGENTA}Moving to home...{RESET}")
            # self.moveit2.move_to_configuration(self.home_joints)
            # self.moveit2.wait_until_executed()
            
            # 3. Move above pick position
            self.get_logger().info(f"{COLOR_MAGENTA}Moving above {instrument}...{RESET}")
            self.moveit2.move_to_pose(
                position=pick_approach,
                quat_xyzw=[pick_quat.x, pick_quat.y, pick_quat.z, pick_quat.w]
            )
            self.moveit2.wait_until_executed()
            
            # 4. Open gripper
            self.get_logger().info(f"{COLOR_MAGENTA}Opening gripper...{RESET}")
            self.gripper.open()
            self.gripper.wait_until_executed()
            
            # 5. Move down to pick
            self.get_logger().info(f"{COLOR_MAGENTA}Moving down to pick {instrument}...{RESET}")
            self.moveit2.move_to_pose(
                position=pick_at_height,
                quat_xyzw=[pick_quat.x, pick_quat.y, pick_quat.z, pick_quat.w],
                cartesian=True
            )
            self.moveit2.wait_until_executed()
            
            # 6. Close gripper to grasp
            self.get_logger().info(f"{COLOR_MAGENTA}Closing gripper to grasp {instrument}...{RESET}")
            self.gripper.close()
            self.gripper.wait_until_executed()
            
            # 7. Lift up
            self.get_logger().info(f"{COLOR_MAGENTA}Lifting {instrument}...{RESET}")
            self.moveit2.move_to_pose(
                position=lift_position,
                quat_xyzw=[pick_quat.x, pick_quat.y, pick_quat.z, pick_quat.w],
                cartesian=True
            )
            self.moveit2.wait_until_executed()

            # --- Place sequence ---
            
            # 8. Move to home
            self.get_logger().info(f"{COLOR_MAGENTA}Moving to home...{RESET}")
            self.moveit2.move_to_configuration(self.home_joints)
            self.moveit2.wait_until_executed()
            
            # 9. Move above place position
            self.get_logger().info(f"{COLOR_MAGENTA}Moving above sorted position...{RESET}")
            self.moveit2.move_to_pose(
                position=place_approach,
                quat_xyzw=[place_quat.x, place_quat.y, place_quat.z, place_quat.w]
            )
            self.moveit2.wait_until_executed()
            
            # 9. Move down to place
            self.get_logger().info(f"{COLOR_MAGENTA}Moving down to place {instrument}...{RESET}")
            self.moveit2.move_to_pose(
                position=place_at_height,
                quat_xyzw=[place_quat.x, place_quat.y, place_quat.z, place_quat.w],
                cartesian=True
            )
            self.moveit2.wait_until_executed()
            
            # 10. Open gripper to release
            self.get_logger().info(f"{COLOR_MAGENTA}Releasing {instrument}...{RESET}")
            self.gripper.open()
            self.gripper.wait_until_executed()
            
            # 11. Lift back up
            self.get_logger().info(f"{COLOR_MAGENTA}Moving back up...{RESET}")
            self.moveit2.move_to_pose(
                position=place_approach,
                quat_xyzw=[place_quat.x, place_quat.y, place_quat.z, place_quat.w],
                cartesian=True
            )
            self.moveit2.wait_until_executed()
            
            # # 13. Close gripper (relaxed)
            # self.gripper.close()
            # self.gripper.wait_until_executed()
            
            # 14. Return to start
            self.get_logger().info(f"{COLOR_MAGENTA}Returning to start position...{RESET}")
            self.moveit2.move_to_configuration(self.start_joints)
            self.moveit2.wait_until_executed()

            # # CRITICAL: Ensure gripper is OPEN at the end
            # self.get_logger().info(f"{COLOR_MAGENTA}Ensuring gripper is open...{RESET}")
            # self.gripper.open()
            # self.gripper.wait_until_executed()
            
            self.get_logger().info(f"{COLOR_GREEN}Pick and place complete for {instrument}{RESET}")
            return True
            
        except Exception as e:
            self.get_logger().error(f"{COLOR_RED}Error in pick_and_place_to_target: {e}{RESET}")
            return False
    
    def pick_only(self, pick_pose: dict, instrument: str):
        """
        Pick instrument from pick_pose and hold (no placement).
        
        Args:
            pick_pose: {'x', 'y', 'z', 'yaw'} - Position and orientation to pick from
            instrument: Name of the instrument
        """
        try:
            pick_quat = self.get_grasp_orientation(pick_pose['yaw'])
            self.get_logger().info(f"{COLOR_BLUE}pick_quat: {pick_quat}{RESET}")
            
            # Calculate positions (transform coordiantes: Gazebo -> Panda frame)
            # (x, y, z) ==> (y, -x, z) ==> Rot 90 CCW
            pick_approach = [
                pick_pose['y'],
                -pick_pose['x'],
                pick_pose['z'] + APPROACH_HEIGHT_OFFSET
            ]
            pick_at_height = [
                pick_pose['y'],
                -pick_pose['x'],
                pick_pose['z'] + 0.11
            ]
            lift_position = [
                pick_pose['y'],
                -pick_pose['x'],
                pick_pose['z'] + APPROACH_HEIGHT_OFFSET
            ]
            drop_position = [
                DROP_ZONE['y'],
                -DROP_ZONE['x'],
                DROP_ZONE['z'],
            ]

            self.get_logger().info(f"{COLOR_BLUE}pick_approach: {pick_approach}{RESET}")
            self.get_logger().info(f"{COLOR_BLUE}pick_at_height: {pick_at_height}{RESET}")
            self.get_logger().info(f"{COLOR_BLUE}lift_position: {lift_position}{RESET}")

            # # 1. Move to start position
            # self.get_logger().info(f"{COLOR_MAGENTA}Moving to start position...{RESET}")
            # self.moveit2.move_to_configuration(self.start_joints)
            # self.moveit2.wait_until_executed()
            
            # # 2. Move to home
            # self.get_logger().info(f"{COLOR_MAGENTA}Moving to home...{RESET}")
            # self.moveit2.move_to_configuration(self.home_joints)
            # self.moveit2.wait_until_executed()
            
            # 3. Move above
            self.get_logger().info(f"{COLOR_MAGENTA}Moving above the instrument{RESET}")
            self.moveit2.move_to_pose(
                position=pick_approach,
                quat_xyzw=[pick_quat.x, pick_quat.y, pick_quat.z, pick_quat.w]
            )
            self.moveit2.wait_until_executed()
            
            # 4. Open gripper
            self.get_logger().info(f"{COLOR_MAGENTA}Opening gripper...{RESET}")
            self.gripper.open()
            self.gripper.wait_until_executed()
            
            # 5. Move down to pick
            self.get_logger().info(f"{COLOR_MAGENTA}Moving down...{RESET}")
            self.moveit2.move_to_pose(
                position=pick_at_height,
                quat_xyzw=[pick_quat.x, pick_quat.y, pick_quat.z, pick_quat.w],
                cartesian=True
            )
            self.moveit2.wait_until_executed()
            
            # 6. Close gripper
            self.get_logger().info(f"{COLOR_MAGENTA}Closing gripper...{RESET}")
            self.gripper.close()
            self.gripper.wait_until_executed()
            
            # 7. Lift up to approach height
            self.get_logger().info(f"{COLOR_MAGENTA}Lifting up...{RESET}")
            self.moveit2.move_to_pose(
                position=pick_approach,
                quat_xyzw=[pick_quat.x, pick_quat.y, pick_quat.z, pick_quat.w],
                cartesian=True
            )
            self.moveit2.wait_until_executed()

            # 7. Move to home
            self.get_logger().info(f"{COLOR_MAGENTA}Moving to home...{RESET}")
            self.moveit2.move_to_configuration(self.home_joints)
            self.moveit2.wait_until_executed()

            # 8. Move to drop zone
            self.get_logger().info(f"{COLOR_MAGENTA}Moving to drop zone...{RESET}")
            self.moveit2.move_to_pose(
                position=drop_position,
                quat_xyzw=[0.0, 1.0, 0.0, 0.0]
            )
            self.moveit2.wait_until_executed()

            # 8. Open gripper to release instrument
            self.get_logger().info(f"{COLOR_MAGENTA}Opening gripper to release {instrument}...{RESET}")
            self.gripper.open()
            self.gripper.wait_until_executed()

            # 10. Move back to start position
            self.get_logger().info(f"{COLOR_MAGENTA}Moving to start position...{RESET}")
            self.moveit2.move_to_configuration(self.start_joints)
            self.moveit2.wait_until_executed()
            
            self.get_logger().info(f"Pick complete for {instrument}, holding in gripper")

            return True
        
        except Exception as e:
            self.get_logger().error(f"{COLOR_RED}Error in pick_only: {e}{RESET}")
            return False
    
    def pick_only_sorted(self, pick_pose: dict, instrument: str):
        """
        Pick instrument from sorted position (already on Table 2).
        Same as pick_only but with different status message.
        """
        result = self.pick_only(pick_pose, instrument)
        if result:
            self.get_logger().info(f"{COLOR_GREEN}Picked {instrument} from sorted table{RESET}")
        return result
    
    def target_callback(self, msg):
        """Process target commands from orchestrator"""

        # Skip if busy
        if self.is_busy:
            self.get_logger().warn(f"Busy processing previous command, ignoring: {msg.data}")
            
            return
        
        # grand_time_start = time.time()
        
        try:
            parts = msg.data.split(',')
            
            if len(parts) < 2:
                self.get_logger().error(f"Invalid command format: {msg.data}")
                return
            
            instrument = parts[0]
            command_type = parts[1]

            # Mark as busy before processing
            self.is_busy = True

            if command_type == 'sort' and len(parts) == 6:
                # Format: instrument,sort,px,py,pz,yaw
                pick_pose = {
                    'x': float(parts[2]),
                    'y': float(parts[3]),
                    'z': float(parts[4]),
                    'yaw': float(parts[5])
                }
                
                self.get_logger().info(
                    f"{COLOR_CYAN}Executing SORT for {instrument} "
                    f"from ({pick_pose['x']:.3f}, {pick_pose['y']:.3f}, yaw:{math.degrees(pick_pose['yaw']):.1f}° / {pick_pose['yaw']}){RESET}"
                )
                
                # Execute sort (pick from current position, move to sorted position)
                start_time = time.time()
                success = self.pick_and_sort(pick_pose, instrument)
                finish_time = time.time()
                print(f"{COLOR_GREEN}Sorting finished in {datetime.timedelta(seconds=finish_time-start_time)}{RESET}")
                
                # Send completion status
                status_data = {
                    "status": "complete" if success else "failed",
                    "instrument": instrument,
                    "operation": "sort"
                }
                self.status_pub.publish(String(data=json.dumps(status_data)))
            
            elif command_type == 'pick_and_place' and len(parts) == 10:
                # Format: instrument,pick_and_place,px,py,pz,yaw,tx,ty,tz,target_yaw
                pick_pose = {
                    'x': float(parts[2]),
                    'y': float(parts[3]),
                    'z': float(parts[4]),
                    'yaw': float(parts[5])
                }
                place_pose = {
                    'x': float(parts[6]),
                    'y': float(parts[7]),
                    'z': float(parts[8]),
                    'yaw': float(parts[9])
                }
                
                self.get_logger().info(
                    f"Executing PICK AND PLACE for {instrument} "
                    f"from ({pick_pose['x']:.3f}, {pick_pose['y']:.3f}, yaw:{math.degrees(pick_pose['yaw']):.1f}°) "
                    f"to ({place_pose['x']:.3f}, {place_pose['y']:.3f}, yaw:{math.degrees(place_pose['yaw']):.1f}°)"
                )
                
                start_time = time.time()
                success = self.pick_and_place_to_target(pick_pose, place_pose, instrument)
                finish_time = time.time()
                print(f"{COLOR_GREEN}Pick and place finished in {datetime.timedelta(seconds=finish_time-start_time)}{RESET}")
                
                # Notify orchestrator of completion
                status_data = {
                    "status": "complete" if success else "failed",
                    "instrument": instrument,
                    "operation": "pick_and_place"
                }
                self.status_pub.publish(String(data=json.dumps(status_data)))
                
            elif command_type == 'pick_only' and len(parts) == 6:
                # Format: instrument,pick_only,px,py,pz,yaw
                pick_pose = {
                    'x': float(parts[2]),
                    'y': float(parts[3]),
                    'z': float(parts[4]),
                    'yaw': float(parts[5])
                }
                
                self.get_logger().info(
                    f"Executing PICK ONLY for {instrument} "
                    f"at ({pick_pose['x']:.3f}, {pick_pose['y']:.3f}, yaw:{math.degrees(pick_pose['yaw']):.1f}°)"
                )

                start_time = time.time()
                success = self.pick_only(pick_pose, instrument)
                finish_time = time.time()
                print(f"{COLOR_GREEN}Picking finished in {datetime.timedelta(seconds=finish_time-start_time)}{RESET}")
                
                # Send completion status
                status_data = {
                    "status": "complete" if success else "failed",
                    "instrument": instrument,
                    "operation": "pick_only"
                }
                self.status_pub.publish(String(data=json.dumps(status_data)))
                
            elif command_type == 'pick_only_sorted' and len(parts) == 6:
                # Format: instrument,pick_only_sorted,px,py,pz,yaw
                pick_pose = {
                    'x': float(parts[2]),
                    'y': float(parts[3]),
                    'z': float(parts[4]),
                    'yaw': float(parts[5])
                }
                
                self.get_logger().info(
                    f"Executing PICK FROM SORTED table for {instrument} "
                    f"at ({pick_pose['x']:.3f}, {pick_pose['y']:.3f}, yaw:{math.degrees(pick_pose['yaw']):.1f}°)"
                )
                
                start_time = time.time()
                success = self.pick_only_sorted(pick_pose, instrument)
                finish_time = time.time()
                print(f"{COLOR_GREEN}Picking (sorted) finished in {datetime.timedelta(seconds=finish_time-start_time)}{RESET}")
                
                status_data = {
                    "status": "complete" if success else "failed",
                    "instrument": instrument,
                    "operation": "pick_only_sorted"
                }
                self.status_pub.publish(String(data=json.dumps(status_data)))

            else:
                self.get_logger().error(
                    f"Unknown command format or wrong number of arguments. "
                    f"Got {len(parts)} parts: {msg.data}"
                )
                status_data = {
                    "status": "error",
                    "error": f"Invalid command format: {msg.data}"
                }
                self.status_pub.publish(String(data=json.dumps(status_data)))
                
        except Exception as e:
            self.get_logger().error(f"Error in target callback: {e}")
            import traceback
            traceback.print_exc()
            status_data = {
                "status": "error",
                "error": str(e)
            }
            self.status_pub.publish(String(data=json.dumps(status_data)))
        
        finally:
            # Always mark as free when done (even on error)
            self.is_busy = False
        
        # grand_time_stop = time.time()
        # print(f"{COLOR_RED}Whole operation finished in {datetime.timedelta(seconds=grand_time_stop - grand_time_start)}{RESET}")

def main(args=None):
    rclpy.init(args=args)
    node = InstrumentPickAndPlace()
    
    # executor = rclpy.executors.MultiThreadedExecutor(2)
    # executor.add_node(node)
    # executor_thread = Thread(target=executor.spin, daemon=True)
    # executor_thread.start()

    # Use a single-threaded executor for simpler handling
    
    try:
        rclpy.spin(node)
        # executor_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()