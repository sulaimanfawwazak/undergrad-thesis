#!/usr/bin/env python3
# panda_vision/panda_vision/yolo_inference.py
import argparse
import cv2
from ultralytics import YOLO
import json
import os
import numpy as np

class_index = {
  0: "needle_holder",
  1: "scalpel",
  2: "scissors",
  3: "tweezers",
  4: "retractor",
}

def get_obb_center(xyxyxyxy):
  corners = np.array(xyxyxyxy).reshape(-1, 2)
  center = np.mean(corners, axis=0)

  return center[0], center[1]

def calculate_distance(point1, point2):
  return np.sqrt(
    (point1[0] - point2[0])**2 + (point1[1] - point2[1]) ** 2
  )

def apply_nms_by_distance(detections, distance_threshold=50, rotation_threshold=0.08):
  if not detections:
    return []
  
  detections = sorted(
    detections, 
    key=lambda d: d["confidence"], 
    reverse=True
  )

  filtered = []

  for det in detections:
    det_center = get_obb_center(det["xyxyxyxy"])
    det_corners = np.array(det["xyxyxyxy"])
    det_rot = det["xywhr"][-1]

    should_keep = True

    for kept in filtered:
      kept_center = get_obb_center(kept["xyxyxyxy"])
      kept_corners = np.array(kept["xyxyxyxy"])
      kept_rot = kept["xywhr"][-1]

      corner_diff = np.mean(
        (det_corners - kept_corners) ** 2
      )

      rot_diff = abs(det_rot - kept_rot)

      if (corner_diff < distance_threshold) and (rot_diff < rotation_threshold):
        should_keep = False
        break

    if should_keep:
      filtered.append(det)

  return filtered


def run_inference(image_path):
  if not os.path.exists(image_path):
    print(json.dumps({"error": f"Image not found: {image_path}"}))
    return []
  
  # model_path = "/home/pwnwas/Personal/College/Skripsi/Code/playground_ws/src/panda_vision/models/yolo26n_736_150.pt"
  # model_path = "/home/pwnwas/Personal/College/Skripsi/Code/playground_ws/src/panda_vision/models/yolo26n_1024_150.pt"
  # model_path = "/home/pwnwas/Personal/College/Skripsi/Code/playground_ws/src/panda_vision/models/yolov8n_736_150.pt"
  # model_path = "/home/pwnwas/Personal/College/Skripsi/Code/playground_ws/src/panda_vision/models/yolov8n_1024_150.pt"
  model_path = "/home/pwnwas/Personal/College/Skripsi/Code/playground_ws/src/panda_vision/models/yolo26n_736_150_621i.pt"
  # model_path = "/home/pwnwas/Personal/College/Skripsi/Code/playground_ws/src/panda_vision/models/yolo26n_1024_150_621i.pt"
  # model_path = "/home/pwnwas/Personal/College/Skripsi/Code/playground_ws/src/panda_vision/models/yolov8n_736_150_621i.pt"
  # model_path = "/home/pwnwas/Personal/College/Skripsi/Code/playground_ws/src/panda_vision/models/yolov8n_1024_150_621i.pt"

  if not os.path.exists(model_path):
    print(json.dumps({"error": f"Model not found: {model_path}"}))
    return []
  
  model = YOLO(model=model_path)
  model.to("cuda")
  
  try:
    results = model.predict(
      image_path,
      task="obb",
      conf=0.2,
      verbose=False,
      device="cuda"
    )

    # Convert results into JSON
    detections = []
    if results[0].obb is not None:
      for obb in results[0].obb:
        xyxyxyxy = obb.xyxyxyxy.cpu().numpy()[0] # Shape (4, 2)
        flattened_coords = xyxyxyxy.flatten().tolist() # [x1, y1, x2, y2, x3, y3, x4, y4]

        detections.append({
          "class_name": class_index.get(int(obb.cls), "unknown"),
          "confidence": float(obb.conf),
          # "xyxyxyxy": obb.xyxyxyxy.cpu().tolist()
          "xyxyxyxy": flattened_coords,
          "xywhr": obb.xywhr.cpu().numpy()[0].tolist()
        })
    
    original_count = len(detections)
    detections = apply_nms_by_distance(detections)

    print(json.dumps(detections))
    return detections

  except Exception as e:
    # print(f"Error while predicting: {e}")
    error_msg = {"error": str(e)}
    print(json.dumps(error_msg))
    return []

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--image", required=True)
  args = parser.parse_args()
  run_inference(args.image)