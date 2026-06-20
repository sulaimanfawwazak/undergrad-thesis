#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import SpawnEntity

def main():
    rclpy.init()
    node = Node('test_spawn_client')
    
    # Try different service names
    service_names = [
        '/world/test/create',
        '/world/default/create', 
        '/world/empty/create',
        '/spawn_entity'
    ]
    
    for service_name in service_names:
        client = node.create_client(SpawnEntity, service_name)
        if client.wait_for_service(timeout_sec=2.0):
            print(f"Found service: {service_name}")
            break
        else:
            print(f"Not found: {service_name}")
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()