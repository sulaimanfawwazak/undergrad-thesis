#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import sys
import time
import subprocess
from threading import Event

class PandaSpawner(Node):
    def __init__(self):
        super().__init__('panda_spawner')
        
        # Declare parameters
        self.declare_parameter('world_name', 'thesis')
        self.declare_parameter('robot_name', 'panda')
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('z', 0.725)
        self.declare_parameter('roll', 0.0)
        self.declare_parameter('pitch', 0.0)
        self.declare_parameter('yaw', 0.0)
        
        self.world_name = self.get_parameter('world_name').value
        self.robot_name = self.get_parameter('robot_name').value
        
        # Wait for robot_description topic
        self.get_logger().info('Waiting for robot_description topic...')
        self.wait_for_topic('/robot_description')
        
        # Wait for spawn service
        self.get_logger().info(f'Waiting for /world/{self.world_name}/create service...')
        self.wait_for_service(f'/world/{self.world_name}/create')
        
        # Spawn with retry
        self.spawn_robot()
    
    def wait_for_topic(self, topic_name, timeout=30):
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = subprocess.run(
                ['ros2', 'topic', 'list'],
                capture_output=True,
                text=True
            )
            if topic_name in result.stdout:
                self.get_logger().info(f'Topic {topic_name} available')
                return True
            time.sleep(1)
        self.get_logger().error(f'Topic {topic_name} not available after {timeout}s')
        return False
    
    def wait_for_service(self, service_name, timeout=30):
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = subprocess.run(
                ['ros2', 'service', 'list'],
                capture_output=True,
                text=True
            )
            if service_name in result.stdout:
                self.get_logger().info(f'Service {service_name} available')
                return True
            time.sleep(1)
        self.get_logger().error(f'Service {service_name} not available after {timeout}s')
        return False
    
    def spawn_robot(self):
        # Get robot_description from topic
        self.get_logger().info('Getting robot description...')
        result = subprocess.run(
            ['ros2', 'topic', 'echo', '--once', '--field', 'data', '/robot_description'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0 or not result.stdout:
            self.get_logger().error('Failed to get robot_description')
            return
        
        robot_description = result.stdout.strip()
        
        # Prepare spawn command
        spawn_cmd = [
            'ros2', 'service', 'call',
            f'/world/{self.world_name}/create',
            'ros_gz_interfaces/srv/SpawnEntity',
            f'{{name: "{self.robot_name}", xml: "{robot_description}"}}'
        ]
        
        # Try multiple times
        for attempt in range(5):
            self.get_logger().info(f'Spawn attempt {attempt + 1}/5...')
            result = subprocess.run(spawn_cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and 'success' in result.stdout.lower():
                self.get_logger().info('Robot spawned successfully!')
                return True
            else:
                self.get_logger().warn(f'Attempt {attempt + 1} failed: {result.stderr}')
                time.sleep(3)
        
        self.get_logger().error('Failed to spawn robot after 5 attempts')
        return False

def main(args=None):
    rclpy.init(args=args)
    spawner = PandaSpawner()
    rclpy.spin_once(spawner, timeout_sec=5.0)
    spawner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()