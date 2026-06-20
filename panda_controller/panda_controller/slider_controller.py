#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class SliderControl(Node):
    def __init__(self):
        super().__init__("slider_control") # Sets the node name "/slider_control"

        # Publisher
        self.arm_pub_ = self.create_publisher(
            JointTrajectory, 
            "arm_controller/joint_trajectory", 
            10
        )
        self.gripper_pub_ = self.create_publisher(
            JointTrajectory, 
            "gripper_controller/joint_trajectory", 
            10
        )
        
        # Subscriber
        self.sub_ = self.create_subscription(
            JointState, 
            "joint_commands", # sub to `/joint_commands`, which is a remapping of `joint_states` from Joint State Publihser GUI slider
            self.sliderCallback, 
            10
        ) # Subscribe to /joint_commands from the slider
        
        self.get_logger().info("Slider Control Node started")

    def sliderCallback(self, msg):
        # Send arm trajectory
        arm_controller = JointTrajectory()
        arm_controller.joint_names = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]
        
        arm_goal = JointTrajectoryPoint()
        arm_goal.positions = list(msg.position[:7]) # Since there are 7 joints, use 7 slider values for each joints
        arm_goal.time_from_start = Duration(seconds=1, nanoseconds=0).to_msg() # Reach target pose in 1 second
        
        arm_controller.points.append(arm_goal)
        arm_controller.header.stamp = self.get_clock().now().to_msg()
        
        # Send gripper trajectory if gripper position is available
        if len(msg.position) > 7:
            gripper_controller = JointTrajectory()
            gripper_controller.joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
            
            gripper_goal = JointTrajectoryPoint()
            gripper_goal.positions = [msg.position[7], msg.position[7]]  # Both fingers move equally
            gripper_goal.time_from_start = Duration(seconds=1, nanoseconds=0).to_msg()
            
            gripper_controller.points.append(gripper_goal)
            gripper_controller.header.stamp = self.get_clock().now().to_msg()
            
            self.gripper_pub_.publish(gripper_controller)
        
        self.arm_pub_.publish(arm_controller)

def main():
    rclpy.init()
    simple_publisher = SliderControl()
    rclpy.spin(simple_publisher)
    simple_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()