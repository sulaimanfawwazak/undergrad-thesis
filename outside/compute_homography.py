import cv2
import numpy as np

needle_holder_yolo = [207.74517822265625, 208.84593200683594]
scissors_yolo = [291.2884216308594, 212.22117614746094]
retractor_yolo = [205.8494873046875, 393.31268310546875]
scalpel_yolo = [292.863525390625, 402.5768127441406]
tweezers_yolo = [383.26556396484375, 395.6221618652344]

needle_holder_yolo_2 = [387.0826416015625, 209.022216796875]
scissors_yolo_2 = [470.3888854980469, 212.29591369628906]
retractor_yolo_2 = [384.176025390625, 392.9676818847656]
scalpel_yolo_2 = [473.01409912109375, 400.5475769042969]
tweezers_yolo_2 = [563.329345703125, 395.5888366699219]

needle_holder_yolo_3 = [747.3429565429688, 209.15390014648438]
scissors_yolo_3 = [448.4173583984375, 213.39425659179688]
retractor_yolo_3 = [744.5147094726562, 392.4467468261719]
scalpel_yolo_3 = [832.8883666992188, 403.71209716796875]
tweezers_yolo_3 = [921.9046630859375, 393.9195251464844]

needle_holder_rl = [-0.48, 0.69]
scissors_rl = [-0.388, 0.69]
retractor_rl = [-0.48, 0.49]
scalpel_rl = [-0.385, 0.49]
tweezers_rl = [-0.285, 0.49]

needle_holder_rl_2 = [-0.28, 0.69]
scissors_rl_2 = [-0.188, 0.69]
retractor_rl_2 = [-0.28, 0.49]
scalpel_rl_2 = [-0.185, 0.49]
tweezers_rl_2 = [-0.085, 0.49]

needle_holder_rl_3 = [0.12, 0.69]
scissors_rl_3 = [-0.212, 0.69]
retractor_rl_3 = [0.12, 0.49]
scalpel_rl_3 = [0.215, 0.49]
tweezers_rl_3 = [0.315, 0.49]

image_pts = np.array([
  needle_holder_yolo,
  scissors_yolo,
  retractor_yolo,
  scalpel_yolo,
  tweezers_yolo,
  needle_holder_yolo_2,
  scissors_yolo_2,
  retractor_yolo_2,
  scalpel_yolo_2,
  tweezers_yolo_2,
  needle_holder_yolo_3,
  scissors_yolo_3,
  retractor_yolo_3,
  scalpel_yolo_3,
  tweezers_yolo_3,
], dtype=np.float32)

world_pts = np.array([
  needle_holder_rl,
  scissors_rl,
  retractor_rl,
  scalpel_rl,
  tweezers_rl,
  needle_holder_rl_2,
  scissors_rl_2,
  retractor_rl_2,
  scalpel_rl_2,
  tweezers_rl_2,
  needle_holder_rl_3,
  scissors_rl_3,
  retractor_rl_3,
  scalpel_rl_3,
  tweezers_rl_3,
], dtype=np.float32)

H, _ = cv2.findHomography(image_pts, world_pts)

print(H)