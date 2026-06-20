#!/usr/bin/env python3

import os
import sys
import random
import math
import time
import subprocess
import argparse
from geometry_msgs.msg import Pose, Quaternion
from ament_index_python.packages import get_package_share_directory
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from charminal import *

instruments_dict = {
    "needle_holder": {
        "bound_x": 0.07185502625,
        "bound_y": 0.1464822884,
        "cx": 0.035927513125,
        "cy": 0.0732411442
    },
    "retractor": {
        "bound_x": 0.09746628189,
        "bound_y": 0.1483472404,
        "cx": 0.048733140945,
        "cy": 0.0741736202,
    },
    "scalpel": {
        "bound_x": 0.01594766235,
        "bound_y": 0.1607148819,
        "cx": 0.007973831175,
        "cy": 0.08035744095
    },
    "scissors": {
        "bound_x": 0.06187911987,
        "bound_y": 0.1317644386,
        "cx": 0.030939559935,
        "cy": 0.0658822193
    },
    "tweezers": {
        "bound_x": 0.02511442566,
        "bound_y": 0.1713867569,
        "cx": 0.01255721283,
        "cy": 0.08569337845
    },
}

table = {
    "length": 0.445,
    "width": 0.340,
    "height": 0.725,
    "cx": -0.23,
    "cy": 0.345,
}

# sorted_positions = {
#     "tweezers":       {"x": 0.13, "y": 0.25, "z": 0.740, "yaw": math.pi/2},
#     "scalpel":      {"x": 0.32, "y": 0.25, "z": 0.740, "yaw": math.pi/2},
#     "scissors":      {"x": 0.08, "y": 0.43, "z": 0.740, "yaw": math.pi},
#     "retractor":     {"x": 0.20, "y": 0.42, "z": 0.740, "yaw": math.pi},
#     "needle_holder": {"x": 0.33, "y": 0.42, "z": 0.740, "yaw": math.pi},
# }

sorted_positions = {
    "tweezers":       {"x": 0.13, "y": 0.46, "z": 0.740, "yaw": math.pi/2},
    "scalpel":      {"x": 0.32, "y": 0.46, "z": 0.740, "yaw": math.pi/2},
    "scissors":      {"x": 0.08, "y": 0.32, "z": 0.740, "yaw": math.pi},
    "retractor":     {"x": 0.20, "y": 0.32, "z": 0.740, "yaw": math.pi},
    "needle_holder": {"x": 0.33, "y": 0.32, "z": 0.740, "yaw": math.pi},
}

"tweezers"
"scalpel"
"scissors"
"retractor"
"needle_holder"

def create_pose(x, y, z, yaw=0.0):
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z

    q = quaternion_from_euler(0, 0, yaw)
    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]

    return pose

def generate_grid_poses(instruments, table, margin=0.03):
    """
    Generate positions in a grid layout instead of a single line.
    This is much more space-efficient and realistic.
    """
    placed_poses = {}
    
    # Shuffle order for random placement
    instrument_list = list(instruments)
    random.shuffle(instrument_list)
    
    # Table boundaries (with padding)
    padding = 0.08
    x_min = table["cx"] - table["length"] / 2 + padding
    x_max = table["cx"] + table["length"] / 2 - padding
    y_min = table["cy"] - table["width"] / 2 + padding
    y_max = table["cy"] + table["width"] / 2 - padding
    
    # Calculate available grid cells
    # Instruments vary in size, so we'll use a dynamic grid based on average instrument size
    avg_width = sum(data["bound_x"] for data in instruments_dict.values()) / len(instruments_dict)
    avg_height = sum(data["bound_y"] for data in instruments_dict.values()) / len(instruments_dict)
    
    # Add margin to cell size
    cell_width = avg_width + margin
    cell_height = avg_height + margin
    
    # Calculate number of rows and columns
    table_width = x_max - x_min
    table_height = y_max - y_min
    
    cols = max(1, int(table_width / cell_width))
    rows = max(1, int(table_height / cell_height))
    
    print(f"[INFO] Grid layout: {rows} rows x {cols} cols")
    print(f"[INFO] Cell size: {cell_width:.3f}m x {cell_height:.3f}m")
    
    # Create list of available grid cells
    grid_cells = []
    for row in range(rows):
        for col in range(cols):
            # Calculate cell center
            x = x_min + (col + 0.5) * cell_width
            y = y_min + (row + 0.5) * cell_height
            
            # Adjust for actual table bounds
            if x_min <= x <= x_max and y_min <= y <= y_max:
                grid_cells.append((x, y))
    
    # Shuffle grid cells for random placement
    random.shuffle(grid_cells)
    
    # Place instruments in grid cells
    for idx, name in enumerate(instrument_list):
        if idx >= len(grid_cells):
            print(f"[WARN] Not enough grid cells for all instruments. {len(instrument_list) - len(grid_cells)} won't fit")
            break
        
        data = instruments_dict[name]
        
        # Random rotation (radians)
        theta = random.uniform(0, 2 * math.pi)
        
        x, y = grid_cells[idx]
        
        # Add slight random offset within cell (jitter)
        x += random.uniform(-margin/2, margin/2)
        y += random.uniform(-margin/2, margin/2)
        
        z = 0.740  # Table surface height
        
        pose = create_pose(x, y, z, theta)
        placed_poses[name] = pose
        
        # Debug print
        roll, pitch, yaw = euler_from_quaternion([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ])
        print(f"[INFO] Placing {name} at cell ({idx//cols}, {idx%cols}) -> "
            f"({x:.3f}, {y:.3f}, yaw: {math.degrees(yaw):.1f}°)")
    
    return placed_poses

def generate_ordered_poses(instruments, table, margin=0.05):
    placed_poses = {}
    table_margin = 0.1

    # Shuffle order
    instrument_list = list(instruments)
    random.shuffle(instrument_list)

    # Table boundaries
    x_min = table["cx"] - table["length"] / 2
    x_max = table["cx"] + table["length"] / 2
    y_min = table["cy"] - table["width"] / 2
    y_max = table["cy"] + table["width"] / 2

    current_x = x_min + table_margin  # start from bottom

    for name in instrument_list:
        data = instruments_dict[name]

        w = data["bound_y"]
        h = data["bound_x"]

        # Random rotation (radians)
        theta = random.uniform(0, 2 * math.pi)

        # Rotated AABB
        w_rot = abs(h * math.sin(theta)) + abs(w * math.cos(theta))
        h_rot = abs(h * math.cos(theta)) + abs(w * math.sin(theta))

        # Y position (random but inside table)
        y = random.uniform(
            y_min + w_rot / 2 + margin,
            y_max - w_rot / 2 - margin
        )

        # X position (stacking)
        x = current_x + h_rot / 2

        # Check overflow
        if x + h_rot / 2 > x_max:
            print(f"[WARN] {name} doesn't fit on table, skipping")
            continue

        # Update for next object
        current_x = x + h_rot / 2 + margin

        # z = table["z"] + data.get("bound_z", 0.005) / 2
        z = 0.740

        pose = create_pose(x, y, z, theta)

        placed_poses[name] = pose

    return placed_poses

def generate_flood_fill_poses(instruments, table, margin=0.005):
    # instruments = ["scissors", "retractor", "needle_holder", "scalpel", "tweezers"]
    placed_poses = {}
    table_margin = 0.1
    margin_x = margin
    margin_y = margin

    # Shuffle order
    instrument_list = list(instruments)
    random.shuffle(instrument_list)

    # Table boundaries
    # (x_min, y_min) -> bottom right
    # (x_max, y_max) -> top left
    padding = 0.01
    # padding = 0.02
    x_min = table["cx"] - table["length"] / 2 + padding
    x_max = table["cx"] + table["length"] / 2 - padding
    y_min = table["cy"] - table["width"] / 2 + padding
    y_max = table["cy"] + table["width"] / 2 - padding

    # Start from bottom left
    current_x = x_min 
    current_y = y_max 

    row_height = 0

    for name in instrument_list:
        print(f"{COLOR_CYAN}Processing pose for: {name}{RESET}")
        data = instruments_dict[name]

        w = data["bound_y"] # Width in Y direction
        h = data["bound_x"] # Height in X direction

        # Random rotation (radians)
        theta = random.uniform(0, 2 * math.pi)
        # if name == "scalpel" or name == "tweezers":
            # theta = math.pi/2
        # else:
            # theta = math.pi

        # Rotated AABB dimension
        w_rot = abs(h * math.sin(theta)) + abs(w * math.cos(theta))
        h_rot = abs(h * math.cos(theta)) + abs(w * math.sin(theta))

        # Check if instrument fits in the current row
        # if current_x + h_rot/2 > x_max:
        if current_x + h_rot > x_max:
            # Move to next row (down/up depending on orientation)
            current_y -= (row_height + margin_y)  # Move down in Y
            current_x = x_min  # Reset X to bottom edge
            row_height = 0  # Reset row height

            # Double-check it fits in Y direction
            if current_y - w_rot < y_min:
                print(f"[WARN] {name} doesn't fit on table (Y overflow), skipping")
                continue

        # Calculate position
        x = current_x + h_rot / 2
        y = current_y - w_rot / 2

        # Update for next instrument
        current_x += h_rot + margin_x
        row_height = max(row_height, w_rot)

        # z = table["z"] + data.get("bound_z", 0.005) / 2
        z = 0.740

        pose = create_pose(x, y, z, theta)
        placed_poses[name] = pose

    return placed_poses

def spawn_model(world_name, model_name, sdf_path, pose):
    with open(sdf_path, 'r') as f:
        sdf = f.read()

    # Clean SDF for CLI
    sdf = sdf.replace('\n', ' ').replace('"', '\\"')

    req = (
        f'name: "{model_name}" '
        f'sdf: "{sdf}" '
        f'pose: {{ position: {{ x: {pose.position.x}, y: {pose.position.y}, z: {pose.position.z} }}, '
        f'orientation: {{ x: {pose.orientation.x}, y: {pose.orientation.y}, z: {pose.orientation.z}, w: {pose.orientation.w} }} }}'
    )

    cmd = [
        'ign', 'service',
        '-s', f'/world/{world_name}/create',
        '--reqtype', 'ignition.msgs.EntityFactory',
        '--reptype', 'ignition.msgs.Boolean',
        '--timeout', '5000',
        '--req', req
    ]

    roll, pitch, yaw = euler_from_quaternion([
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ])
    roll = math.degrees(roll)
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)

    print(f"[INFO] Spawning {model_name} at ({pose.position.x}, {pose.position.y}, {pose.position.z}, {roll}, {pitch}, {yaw})")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"[DEBUG] stdout: {result.stdout}")
        print(f"[OK] {model_name} spawned")
        return True
    else:
        print(f"[ERROR] {model_name} failed:\n{result.stderr}")
        return False

def remove_model(world_name, model_name):
    """
    Remove a model from Gazebo simulation.
    
    Args:
        world_name (str): Name of the Gazebo world
        model_name (str): Name of the model to remove
    
    Returns:
        bool: True if successful, False otherwise
    """
    req = f'name: "{model_name}"'
    
    cmd = [
        'ign', 'service',
        '-s', f'/world/{world_name}/remove',
        '--reqtype', 'ignition.msgs.Entity',
        '--reptype', 'ignition.msgs.Boolean',
        '--timeout', '5000',
        '--req', req
    ]
    
    print(f"[INFO] Removing model: {model_name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"[OK] {model_name} removed")
        return True
    else:
        print(f"[ERROR] Failed to remove {model_name}:\n{result.stderr}")
        return False

def remove_all_instruments(world_name, instrument_names):
    """
    Remove all instruments from the simulation.
    Useful for resetting before sorting.
    
    Args:
        world_name (str): Name of the Gazebo world
        instrument_names (list): List of instrument model names
    
    Returns:
        int: Number of successfully removed instruments
    """
    success_count = 0
    for name in instrument_names:
        if remove_model(world_name, name):
            success_count += 1
        time.sleep(0.2)  # Small delay between removals
    
    print(f"[INFO] Removed {success_count}/{len(instrument_names)} instruments")
    return success_count

def respawn_instrument(world_name, model_name, sdf_path, new_pose):
    """
    Move an instrument by removing and re-spawning it.
    Useful for sorting operations.
    
    Args:
        world_name (str): Name of the Gazebo world
        model_name (str): Name of the model to move
        sdf_path (str): Path to model SDF file
        new_pose (Pose): New pose for the instrument
    
    Returns:
        bool: True if successful, False otherwise
    """
    if remove_model(world_name, model_name):
        time.sleep(0.3)  # Wait for removal to process
        return spawn_model(world_name, model_name, sdf_path, new_pose)
    return False

def wait_for_gz_service(world_name, timeout=30):
    service = f"/world/{world_name}/create"

    start = time.time()

    while time.time() - start < timeout:
        result = subprocess.run(
            ["ign", "service", "-l"],
            capture_output=True,
            text=True
        )

        if service in result.stdout:
            print(f"[OK] Found {service}")
            return True

        time.sleep(1)

    return False

def parse_arguments():
    """Parse command line arguments in ROS2 launch style"""
    args = {
        'world_name': 'test',
        'randomize': 'false',
        'random_seed': '42',
        'layout': 'flood'  # New parameter: 'grid' or 'line'
    }
    
    for arg in sys.argv[1:]:
        if ':=' in arg:
            key, value = arg.split(':=', 1)
            key = key.strip()
            value = value.strip()
            
            if key == 'world_name':
                args['world_name'] = value
            elif key == 'randomize':
                args['randomize'] = value.lower()
            elif key == 'random_seed':
                args['random_seed'] = value
            elif key == 'layout':
                args['layout'] = value.lower()
    
    args['randomize'] = args['randomize'] in ['true', '1', 'yes']
    
    return args

def main():
    cli_args = parse_arguments()
    print(cli_args)
    
    world_name = cli_args['world_name']
    randomize = cli_args['randomize']
    random_seed = int(cli_args['random_seed'])
    layout = cli_args['layout']

    instruments = {
        'needle_holder': 'needle_holder',
        'retractor': 'retractor',
        'scalpel': 'scalpel',
        'scissors': 'scissors',
        'tweezers': 'tweezers'
    }

    if randomize:
        random.seed(random_seed)
        
        # Choose layout algorithm
        if layout == 'flood':
            print(f"[INFO] Using FLOOD layout with seed {random_seed}")
            poses = generate_flood_fill_poses(instruments.keys(), table)
        else:
            print(f"[INFO] Using LINE layout with seed {random_seed}")
            poses = generate_ordered_poses(instruments.keys(), table)

    pkg_path = get_package_share_directory('panda_description')
    models_dir = os.path.join(pkg_path, 'models')
    
    print(f"Waiting for Gazebo (world: {world_name})...")
    
    if not wait_for_gz_service(world_name):
        print(f"[ERROR] Gazebo service for world '{world_name}' not available")
        return

    success = 0

    for name, folder in instruments.items():
        sdf_path = os.path.join(models_dir, folder, 'model.sdf')

        if not os.path.exists(sdf_path):
            print(f"[ERROR] Missing {sdf_path}")
            continue

        if randomize:
            if name not in poses:
                print(f"[WARN] {name} not in generated poses, skipping")
                continue
            pose = poses[name]
        else:
            x = sorted_positions[name]["x"]
            y = sorted_positions[name]["y"]
            z = sorted_positions[name]["z"]
            yaw = sorted_positions[name]["yaw"]
            pose = create_pose(x, y, z, yaw)

        if spawn_model(world_name, name, sdf_path, pose):
            success += 1
        
        time.sleep(0.2)

    print(f"[INFO] Successfully spawned {success}/{len(instruments)} instruments")

    # Example usage of remove functions (commented out for safety)
    # To remove a single instrument:
    # remove_model(world_name, "scalpel")
    
    # To remove all instruments:
    # remove_all_instruments(world_name, spawned_models)

if __name__ == "__main__":
    main()