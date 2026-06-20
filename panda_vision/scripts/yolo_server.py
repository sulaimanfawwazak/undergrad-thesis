#!/home/pwnwas/miniconda3/envs/colab_mimic/bin/python
# panda_vision/scripts/yolo_server.py
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
            conf=0.2,
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

        for det in detections:
            det_corners = np.array(det["xyxyxyxy"])
            det_rot = det["xywhr"][-1]
    
            keep = True
    
            for kept in filtered:
                kept_corners = np.array(kept["xyxyxyxy"])
                kept_rot = kept["xywhr"][-1]

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
    model_path = "/home/pwnwas/Personal/College/Skripsi/Code/playground_ws/src/panda_vision/models/yolo26n_736_150_621i.pt"
    server = YOLOServer(model_path)
    server.run()