#!/usr/bin/env python3
# panda_description/src/spawn_instruments.py
import sys
import random
import argparse
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose, Point, Quaternion
from os import pathsep
from pathlib import Path
import math

# Set resource path for models
model_path = str(Path(panda_description).parent.resolve()) # panda_description/share/panda_description -> panda_description/share/
model_path += pathsep + os.path.join(get_package_share_directory("panda_description"), 'models') # panda_description/share/models


class InstrumentSpawner(Node):
    def __init__(self):
        super().__init__('instrument_spawner')
        self.client = self.create_client(SpawnEntity, '/spawn_entity')
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for spawn_entity service...')

        self.needleholder_boundbox = {
            "x": 0.07185502625,
            "y": 0.1464822884,
            "z": 0.002954005241,
        }

        self.retractor_boundbox = {
            "x": 0.09746628189,
            "y": 0.1483472404,
            "z": 0.01124949551,
        }

        self.needleholder_xml_ = f"""
        <model name={name}>
        <pose>{pose_x} {pose_y} {pose_z} 0 0 {pose_yaw}</pose>
        <static>false</static>
        <self_collide>false</self_collide>

            <link name='link'>
            <inertial>
                <mass>{needleholder_mass}</mass>
                <inertia>
                <ixx>{needleholder_ixx}</ixx>
                <ixy>{needleholder_ixy}</ixy>
                <ixz>{needleholder_ixz}</ixz>
                <iyy>{needleholder_iyy}</iyy>
                <iyz>{needleholder_iyz}</iyz>
                <izz>{needleholder_izz}</izz>
                </inertia>
            </inertial>

            <collision name='collision'>
                <geometry>
                <mesh>
                    <uri>model://thesis/DAE/NeedleHolder/NeedleHolderCollision.dae</uri>
                    <scale>1 1 1</scale>
                </mesh>
                </geometry>
                <surface>
                <contact>
                    <ode>
                    <max_vel>0.01</max_vel>
                    </ode>
                </contact>
                </surface>
            </collision>

            <visual name='visual'>
                <geometry>
                <mesh>
                    <uri>model://thesis/DAE/NeedleHolder/NeedleHolder.dae</uri>
                </mesh>
                </geometry>
                <meta>
                <layer>1</layer>
                </meta>
            </visual>

            </link>
        </model>
        """

        self.retractor_xml_ = f"""
        <model name={name}>
        <pose>{pose_x} {pose_y} {pose_z} 0 0 {pose_yaw}</pose>
        <self_collide>false</self_collide>
        <static>false</static>

            <link name='link'>
            <inertial>
                <mass>{retractor_mass}</mass>
                <inertia>
                <ixx>{retractor_ixx}</ixx>
                <ixy>{retractor_ixy}</ixy>
                <ixz>{retractor_ixz}</ixz>
                <iyy>{retractor_iyy}</iyy>
                <iyz>{retractor_iyz}</iyz>
                <izz>{retractor_izz}</izz>
                </inertia>
            </inertial>

            <collision name='collision'>
                <geometry>
                <mesh>
                    <uri>model://thesis/DAE/Retractor/RetractorCollision.dae</uri>
                    <scale>1 1 1</scale>
                </mesh>
                </geometry>
                <surface>
                <contact>
                    <ode>
                    <max_vel>0.01</max_vel>
                    </ode>
                </contact>
                </surface>
            </collision>

            <visual name='visual'>
                <geometry>
                <mesh>
                    <uri>model://thesis/DAE/Retractor/Retractor.dae</uri>
                </mesh>
                </geometry>
                <meta>
                <layer>1</layer>
                </meta>
            </visual>

            </link>
        </model>
        """
    
        self.instrument_dict_ = {
            "needleholder": self.needleholder_xml_,
            "retractor": self.retractor_xml_,
        }
    
    # def spawn_instrument(self, name, model_path, pose, static=False):
    #     """Spawn a single instrument"""
    #     request = SpawnEntity.Request()
    #     request.name = name
    #     request.xml = f"""
    #     <model name='{name}'>
    #         <static>{str(static).lower()}</static>
    #         <link name='link'>
    #             <inertial>
    #                 <!-- You can load mass/inertia from your xacro variables -->
    #             </inertial>
    #             <visual name='visual'>
    #                 <geometry>
    #                     <mesh>
    #                         <uri>{model_path}</uri>
    #                     </mesh>
    #                 </geometry>
    #             </visual>
    #             <collision name='collision'>
    #                 <geometry>
    #                     <mesh>
    #                         <uri>{model_path.replace('.dae', 'Collision.dae')}</uri>
    #                     </mesh>
    #                 </geometry>
    #             </collision>
    #         </link>
    #         <pose>{pose.position.x} {pose.position.y} {pose.position.z} 
    #               {pose.orientation.x} {pose.orientation.y} 
    #               {pose.orientation.z} {pose.orientation.w}</pose>
    #     </model>
    #     """
    #     request.initial_pose = pose
        
    #     future = self.client.call_async(request)
    #     rclpy.spin_until_future_complete(self, future)
        
    #     if future.result() is not None:
    #         self.get_logger().info(f'Successfully spawned {name}')
    #     else:
    #         self.get_logger().error(f'Failed to spawn {name}')
        
    #     return future.result()
    
    def spawn_instrument_dev(self, instrument, name, model_path, pose, static=False):
        """Spawn a single instrument"""

        request = SpawnEntity.Request()
        request.name = name
        request.xml = self.instrument_dict_[instrument]
        
        request.initial_pose = pose
        
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            self.get_logger().info(f'Successfully spawned {name}')
        else:
            self.get_logger().error(f'Failed to spawn {name}')
        
        return future.result()

# def random_pose_on_table(table_center_x, table_center_y, table_z, table_width, table_depth, random_rotation=True):
#     """Generate random pose on table surface"""
#     pose = Pose()
    
#     # Random position within table bounds (with margin)
#     margin = 0.05  # 5cm from edges
#     pose.position.x = random.uniform(
#         table_center_x - table_width/2 + margin,
#         table_center_x + table_width/2 - margin
#     )
#     pose.position.y = random.uniform(
#         table_center_y - table_depth/2 + margin,
#         table_center_y + table_depth/2 - margin
#     )
#     pose.position.z = table_z  # Table surface height
    
#     # Random rotation around Z axis (yaw)
#     if random_rotation:
#         yaw = random.uniform(0, 2 * math.pi)
#         pose.orientation = Quaternion(
#             x=0.0, y=0.0, 
#             z=math.sin(yaw/2), 
#             w=math.cos(yaw/2)
#         )
#     else:
#         pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    
#     return pose

def random_pose_on_table_dev():
    """
    1. Take the bounding box x, y (`self.needleholder_boundbox["x"]` & `self.needleholder_boundbox["y"]`)
    2. Rotate the bounding box with an angle a
    3. Find the bounding box of the new rotated-bounding-box (using min(x1, x2, x3, x4), min(y1, y2, y3, y4), max(x1, x2, x3, x4), and max(y1, y2, y3, y4))
    4. Place the next instrument outside the new bounding box
    """
    pass

def main(args=None):
    rclpy.init(args=args)
    
    # Define instrument configurations
    instruments = {
        'needle_holder': {
            'model_path': 'model://thesis/DAE/NeedleHolder/NeedleHolder.dae',
            'mass': 0.02683,  # from your xacro
            'default_pose': {'x': 0.0, 'y': 0.214, 'z': 0.8, 'yaw': 0.0}
        },
        'retractor': {
            'model_path': 'model://thesis/DAE/Retractor/Retractor.dae',
            'mass': 0.04748,
            'default_pose': {'x': 0.02, 'y': 0.214, 'z': 0.8, 'yaw': 0.0}
        },
        'scalpel': {
            'model_path': 'model://thesis/DAE/Scalpel/Scalpel.dae',
            'mass': 0.01688,
            'default_pose': {'x': 0.02, 'y': 0.294, 'z': 0.8, 'yaw': 0.0}
        },
        'scissors': {
            'model_path': 'model://thesis/DAE/Scissors/Scissors.dae',
            'mass': 0.02560,
            'default_pose': {'x': 0.04, 'y': 0.294, 'z': 0.8, 'yaw': 0.0}
        },
        'tweezers': {
            'model_path': 'model://thesis/DAE/Tweezers/Tweezers.dae',
            'mass': 0.03366,
            'default_pose': {'x': 0.374, 'y': 0.194, 'z': 0.8, 'yaw': 0.0}
        }
    }
    
    # Parse arguments (can be passed via launch file)
    randomize = False
    random_seed = 42
    
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if 'randomize:=true' in arg:
                randomize = True
            elif 'random_seed:=' in arg:
                random_seed = int(arg.split(':=')[1])
    
    if randomize:
        random.seed(random_seed)
        print(f"Randomizing instrument positions with seed {random_seed}")
    
    spawner = InstrumentSpawner()
    
    # Spawn each instrument
    for name, config in instruments.items():
        if randomize:
            # Spawn on table surface (adjust these values for your table)
            # Table 1 is at y = 0.505, size 1.2 x 0.7
            pose = random_pose_on_table(
                table_center_x=0.0,
                table_center_y=0.505,  # Table 1 position
                table_z=0.8,  # Table surface height
                table_width=1.2,
                table_depth=0.7,
                random_rotation=True
            )
            print(f"Spawning {name} at random pose: ({pose.position.x}, {pose.position.y}, {pose.position.z})")
        else:
            # Use fixed poses from your world file
            pose = Pose()
            pose.position.x = config['default_pose']['x']
            pose.position.y = config['default_pose']['y']
            pose.position.z = config['default_pose']['z']
            yaw = config['default_pose']['yaw']
            pose.orientation = Quaternion(
                x=0.0, y=0.0,
                z=math.sin(yaw/2),
                w=math.cos(yaw/2)
            )
        
        spawner.spawn_instrument(name, config['model_path'], pose)
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()