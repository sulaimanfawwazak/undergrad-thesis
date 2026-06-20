#!/usr/bin/env python3

import os
import time
import subprocess

from geometry_msgs.msg import Pose
from ament_index_python.packages import get_package_share_directory
from tf_transformations import quaternion_from_euler


# ============================================================
# TABLE PARAMETERS
# ============================================================

TABLE = {
    "length": 1.35,
    "width": 1.18,
    "height": 0.725,
    "cx": 0.0,
    # "cy": 1.175,
    "cy": 0.675,
}


# ============================================================
# BOX PARAMETERS
# ============================================================

BOX = {
    "length": 0.05,
    "width": 0.05,
    "height": 0.02,
}

BOX_MARGIN = 0.05

# If your mesh origin is at the BASE:
# BOX_Z = TABLE["height"]

# If your mesh origin is at the CENTER:
BOX_Z = TABLE["height"] + BOX["height"] / 2


# ============================================================
# UTILITIES
# ============================================================

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


# ============================================================
# GRID GENERATION
# ============================================================

def generate_grid_poses():

    pitch_x = BOX["length"] + BOX_MARGIN
    pitch_y = BOX["width"] + BOX_MARGIN

    cols = int(TABLE["length"] // pitch_x) - 1
    rows = int(TABLE["width"] // pitch_y) - 4

    print(f"[INFO] Grid size: {rows} rows x {cols} cols")
    print(f"[INFO] Total boxes: {rows * cols}")

    x_min = TABLE["cx"] - TABLE["length"] / 2 + 0.05
    y_min = TABLE["cy"] - TABLE["width"] / 2 + 0.1

    poses = []

    idx = 0

    for row in range(rows):
        for col in range(cols):

            x = x_min + pitch_x / 2 + col * pitch_x
            y = y_min + pitch_y / 2 + row * pitch_y

            pose = create_pose(
                x=x,
                y=y,
                z=BOX_Z,
                yaw=0.0
            )

            model_name = f"box_{idx:03d}"

            poses.append(
                (
                    model_name,
                    pose
                )
            )

            idx += 1

    return poses


# ============================================================
# SPAWNING
# ============================================================

def spawn_model(world_name, model_name, sdf_path, pose):

    with open(sdf_path, "r") as f:
        sdf = f.read()

    sdf = sdf.replace("\n", " ").replace('"', '\\"')

    req = (
        f'name: "{model_name}" '
        f'sdf: "{sdf}" '
        f'pose: {{ '
        f'position: {{ '
        f'x: {pose.position.x}, '
        f'y: {pose.position.y}, '
        f'z: {pose.position.z} '
        f'}}, '
        f'orientation: {{ '
        f'x: {pose.orientation.x}, '
        f'y: {pose.orientation.y}, '
        f'z: {pose.orientation.z}, '
        f'w: {pose.orientation.w} '
        f'}} '
        f'}}'
    )

    cmd = [
        "ign",
        "service",
        "-s",
        f"/world/{world_name}/create",
        "--reqtype",
        "ignition.msgs.EntityFactory",
        "--reptype",
        "ignition.msgs.Boolean",
        "--timeout",
        "5000",
        "--req",
        req,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(
            f"[OK] {model_name} "
            f"({pose.position.x:.3f}, "
            f"{pose.position.y:.3f}, "
            f"{pose.position.z:.3f})"
        )
        return True

    print(
        f"[ERROR] Failed to spawn {model_name}\n"
        f"{result.stderr}"
    )
    return False


# ============================================================
# MAIN
# ============================================================

def main():

    world_name = "homography"

    print(
        f"Waiting for Gazebo world '{world_name}'..."
    )

    if not wait_for_gz_service(world_name):
        print("[ERROR] Gazebo not available")
        return

    pkg_path = get_package_share_directory(
        "panda_description"
    )

    models_dir = os.path.join(
        pkg_path,
        "models"
    )

    sdf_path = os.path.join(
        models_dir,
        "HomoBoxRed",
        "model.sdf"
    )

    if not os.path.exists(sdf_path):
        print(f"[ERROR] Missing {sdf_path}")
        return

    poses = generate_grid_poses()
    print(poses)

    success = 0

    for model_name, pose in poses:

        if spawn_model(
            world_name,
            model_name,
            sdf_path,
            pose
        ):
            success += 1

        time.sleep(0.02)

    print(
        f"[INFO] Spawned "
        f"{success}/{len(poses)} boxes"
    )


if __name__ == "__main__":
    main()