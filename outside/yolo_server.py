#!/home/pwnwas/miniconda3/envs/colab_mimic/bin/python
# outside/yolo_server.py
import cv2
import numpy as np
from ultralytics import YOLO
import json
import zmq
import base64
import signal
import sys
import numpy as np
from charminal import *
import math

class YOLOServer:
    def __init__(self, model_path, port=5555):
        self.model = YOLO(model_path)
        self.model.to("cuda")
        
        # ZeroMQ setup
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        
        print(f"YOLO Server running on port {port}")
        print(f"Model loaded: {model_path}")
        
    def process_image(self, image_data):
        """Process image and return detections"""
        # Decode image from bytes
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Run inference
        results = self.model.predict(
            frame,
            task="obb",
            conf=0.5,
            verbose=False,
            device="cuda"
        )
        
        # Process results
        detections = []
        if results[0].obb is not None:
            for obb in results[0].obb:
                xyxyxyxy = obb.xyxyxyxy.cpu().numpy()[0]
                flattened_coords = xyxyxyxy.flatten().tolist()
                
                detections.append({
                    "class_name": self.get_class_name(int(obb.cls)),
                    "confidence": float(obb.conf),
                    "xyxyxyxy": flattened_coords,
                    "xywhr": obb.xywhr.cpu().numpy()[0].tolist()
                })
        
        # Apply NMS
        detections = self.apply_nms_by_distance(detections)
        
        return detections
    
    def get_class_name(self, class_id):
        class_map = {
            0: "needle_holder",
            1: "scalpel",
            2: "scissors",
            3: "tweezers",
            4: "retractor",
        }
        return class_map.get(class_id, "unknown")
    
    # def apply_nms(self, detections, iou_threshold=0.5):
    #     """Apply NMS (simplified)"""
    #     if not detections:
    #         return []
        
    #     detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    #     filtered = []
        
    #     for det in detections:
    #         keep = True
    #         for kept in filtered:
    #             if self.calculate_iou(det['xyxyxyxy'], kept['xyxyxyxy']) > iou_threshold:
    #                 keep = False
    #                 break
    #         if keep:
    #             filtered.append(det)
        
    #     return filtered
    
    def calculate_distance(self, point1, point2):
        return np.sqrt(
          (point1[0] - point2[0])**2 + (point1[1] - point2[1]) ** 2
        )
    
    def apply_nms_by_distance(self, detections, distance_threshold=50, rotation_threshold=0.08):
        if not detections:
            return []
        
        detections = sorted(
            detections, 
            key=lambda d: d["confidence"], 
            reverse=True
        )

        filtered = []

        # Calculate the rotation based on the length (longest line between x1 and x2, or x1 and x3)
        for det in detections:
            x1, y1, x2, y2, x3, y3, x4, y4 = det["xyxyxyxy"]

            y_min = min(y1, y2, y3, y4)

            if y_min == y1:
                # Find distance between x1-x2 and x1-x4
                dist_1 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                dist_2 = np.sqrt((x1 - x4)**2 + (y1 - y4)**2)

                if dist_1 > dist_2:
                    dx = x2 - x1
                    dy = y2 - y1
                    det_rot = np.arctan2(dy, dx)
                else:
                    dx = x4 - x1
                    dy = y4 - y1
                    det_rot = np.arctan2(dy, dx)

            elif y_min == y2:
                # Find distance between x2-x1 and x2-x3
                dist_1 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                dist_2 = np.sqrt((x2 - x3)**2 + (y2 - y3)**2)

                if dist_1 > dist_2:
                    dx = x1 - x2
                    dy = y1 - y2
                    det_rot = np.arctan2(dy, dx)
                else:
                    dx = x3 - x2
                    dy = y3 - y2
                    det_rot = np.arctan2(dy, dx)

            elif y_min == y3:
                # Find distance between x3-x2 and x3-x4
                dist_1 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
                dist_2 = np.sqrt((x3 - x4)**2 + (y3 - y4)**2)

                if dist_1 > dist_2:
                    dx = x2 - x3
                    dy = y2 - y3
                    det_rot = np.arctan2(dy, dx)
                else:
                    dx = x4 - x3
                    dy = y4 - y3
                    det_rot = np.arctan2(dy, dx)

            elif y_min == y4:
                # Find distance between x4-x1 and x4-x3
                dist_1 = np.sqrt((x4 - x1)**2 + (y4 - y1)**2)
                dist_2 = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)

                if dist_1 > dist_2:
                    dx = x1 - x4
                    dy = y1 - y4
                    det_rot = np.arctan2(dy, dx)
                else:
                    dx = x3 - x4
                    dy = y3 - y4
                    det_rot = np.arctan2(dy, dx)

            det_rot = -1 * det_rot

            det_corners = np.array(det["xyxyxyxy"])
            # det_rot = det["xywhr"][-1]
    
            keep = True
    
            for kept in filtered:
                kx1, ky1, kx2, ky2, kx3, ky3, kx4, ky4 = kept["xyxyxyxy"]

                ky_min = min(ky1, ky2, ky3, ky4)

                if ky_min == ky1:
                    # Find distance between x1-x2 and x1-x4
                    dist_1 = np.sqrt((kx1 - kx2)**2 + (ky1 - ky2)**2)
                    dist_2 = np.sqrt((kx1 - kx4)**2 + (ky1 - ky4)**2)

                    if dist_1 > dist_2:
                        kdx = kx2 - kx1
                        kdy = ky2 - ky1
                        kept_rot = np.arctan2(kdy, kdx)
                    else:
                        kdx = kx4 - kx1
                        kdy = ky4 - ky1
                        kept_rot = np.arctan2(kdy, kdx)

                elif ky_min == ky2:
                    # Find distance between x2-x1 and x2-x3
                    dist_1 = np.sqrt((kx2 - kx1)**2 + (ky2 - ky1)**2)
                    dist_2 = np.sqrt((kx2 - kx3)**2 + (ky2 - ky3)**2)

                    if dist_1 > dist_2:
                        kdx = kx1 - kx2
                        kdy = ky1 - ky2
                        kept_rot = np.arctan2(kdy, kdx)
                    else:
                        kdx = kx3 - kx2
                        kdy = ky3 - ky2
                        kept_rot = np.arctan2(kdy, kdx)

                elif ky_min == ky3:
                    # Find distance between x3-x2 and x3-x4
                    dist_1 = np.sqrt((kx3 - kx2)**2 + (ky3 - ky2)**2)
                    dist_2 = np.sqrt((kx3 - kx4)**2 + (ky3 - ky4)**2)

                    if dist_1 > dist_2:
                        kdx = kx2 - kx3
                        kdy = ky2 - ky3
                        kept_rot = np.arctan2(kdy, kdx)
                    else:
                        kdx = kx4 - kx3
                        kdy = ky4 - ky3
                        kept_rot = np.arctan2(kdy, kdx)

                elif ky_min == ky4:
                    # Find distance between x4-x1 and x4-x3
                    dist_1 = np.sqrt((kx4 - kx1)**2 + (ky4 - ky1)**2)
                    dist_2 = np.sqrt((kx4 - kx3)**2 + (ky4 - ky3)**2)

                    if dist_1 > dist_2:
                        kdx = kx1 - kx4
                        kdy = ky1 - ky4
                        kept_rot = np.arctan2(kdy, kdx)
                    else:
                        kdx = kx3 - kx4
                        kdy = ky3 - ky4
                        kept_rot = np.arctan2(kdy, kdx)

                kept_rot = -1 * kept_rot

                kept_corners = np.array(kept["xyxyxyxy"])
                # kept_rot = kept["xywhr"][-1]

                corner_diff = np.mean(
                (det_corners - kept_corners) ** 2
                )

                rot_diff = abs(det_rot - kept_rot)

                if (corner_diff < distance_threshold) and (rot_diff < rotation_threshold):
                    keep = False
                    break
    
            if keep:
                filtered.append(det)

        print(f"{COLOR_CYAN}[pwn] filtered: {filtered}{RESET}")
        return filtered
    
    # def calculate_iou(self, box1, box2):
    #     """Simple IoU for axis-aligned (fast)"""
    #     # Convert to axis-aligned for speed
    #     box1_arr = np.array(box1).reshape(-1, 2)
    #     box2_arr = np.array(box2).reshape(-1, 2)
        
    #     x1_min, x1_max = box1_arr[:, 0].min(), box1_arr[:, 0].max()
    #     y1_min, y1_max = box1_arr[:, 1].min(), box1_arr[:, 1].max()
    #     x2_min, x2_max = box2_arr[:, 0].min(), box2_arr[:, 0].max()
    #     y2_min, y2_max = box2_arr[:, 1].min(), box2_arr[:, 1].max()
        
    #     # Intersection
    #     x_left = max(x1_min, x2_min)
    #     y_top = max(y1_min, y2_min)
    #     x_right = min(x1_max, x2_max)
    #     y_bottom = min(y1_max, y2_max)
        
    #     if x_right < x_left or y_bottom < y_top:
    #         return 0.0
        
    #     intersection = (x_right - x_left) * (y_bottom - y_top)
    #     area1 = (x1_max - x1_min) * (y1_max - y1_min)
    #     area2 = (x2_max - x2_min) * (y2_max - y2_min)
    #     union = area1 + area2 - intersection
        
    #     return intersection / union if union > 0 else 0.0
    
    def run(self):
        while True:
            try:
                message = self.socket.recv()
                request = json.loads(message)
                
                if request['type'] == 'inference':
                    # Decode base64 image
                    image_data = base64.b64decode(request['image'])
                    detections = self.process_image(image_data)
                    
                    # Send response
                    response = json.dumps({
                        'success': True,
                        'detections': detections
                    })
                    self.socket.send(response.encode())
                    
            except Exception as e:
                error_response = json.dumps({
                    'success': False,
                    'error': str(e)
                })
                self.socket.send(error_response.encode())

if __name__ == '__main__':
    model_path = "/home/pwnwas/Personal/College/Skripsi/Code/playground_ws/src/panda_vision/models/yolo26n_736_150_669i.pt"
    server = YOLOServer(model_path)
    server.run()