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
from charminal import *
import speech_recognition as sr
import pyaudio
import threading
import time
import tempfile
import wave
import io
import os
import random

mock = True

# test_commands.py - Use this to generate test phrases
test_commands = {
    "scalpel": [
        "hand me the scalpel",
        "give me the scalpel", 
        "i need to make an incision",
        "time to cut",
        "scalpel please",
        "making the incision now",
        "i'll incise here",
        "pass the blade",
        "need to cut",
        "starting the incision"
    ],
    
    "scissors": [
        "hand me the scissors",
        "give me the scissors",
        "cut this suture",
        "scissors please",
        "time to snip",
        "need to cut the stitch",
        "i need to snip this",
        "cutting the ligature",
        "pass the scissors",
        "i need to dissect here"
    ],
    
    "needle_holder": [
        "hand me the needle holder",
        "give me the needle driver",
        "ready to suture",
        "needle holder please",
        "time to close this up",
        "i need to stitch this",
        "pass the needle driver",
        "starting the sutures",
        "closing the incision",
        "i'll suture now"
    ],
    
    "retractor": [
        "hand me the retractor",
        "give me the retractor",
        "i need better exposure",
        "retractor please",
        "let me retract this",
        "need to expose the site",
        "pull this back",
        "hold this open",
        "i need to see better",
        "expose the area please"
    ],
    
    "tweezers": [
        "hand me the tweezers",
        "give me the forceps",
        "i need to grasp this",
        "tweezers please",
        "let me pick this up",
        "need finer control",
        "forceps please",
        "grasp this tissue",
        "pickups please",
        "i'll manipulate this"
    ]
}

class YOLOServer:
    def __init__(self, model_path, port=5555):
        self.model = YOLO(model_path)
        self.model.to("cuda")
        
        # ZeroMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")

        # Speech recognition setup
        self.speech_recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()  # Using default mic
        
        # Store the audio source context properly
        self.microphone_source = None
        self.recording_lock = threading.Lock()
        
        # Calibrate for ambient noise (do this once at startup)
        print("Calibrating microphone for ambient noise...")
        with self.microphone as source:
            self.speech_recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Microphone calibrated")

        # For push-to-talk recording
        self.is_recording = False
        self.recording_audio_data = []
        self.recording_thread = None
        self.stop_recording_event = threading.Event()
        
        print(f"YOLO Server running on port {port}")
        print(f"Model loaded: {model_path}")
        print("Speech recognition ready for push-to-talk")

    def start_recording(self):
        """Start recording audio (push-to-talk press)"""
        with self.recording_lock:
            if self.is_recording:
                return {"success": False, "error": "Already recording"}
            
            self.is_recording = True
            self.recording_audio_data = []
            self.stop_recording_event.clear()
            
            # Start recording in background thread
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            print("Recording started...")
            return {"success": True, "message": "Recording started"}

    def _record_audio(self):
        """Background recording thread - opens mic once and records chunks"""
        try:
            # Open microphone context once for the entire recording session
            with self.microphone as source:
                # Small adjustment each session (optional)
                self.speech_recognizer.adjust_for_ambient_noise(source, duration=0.3)
                
                # Record until told to stop
                while self.is_recording and not self.stop_recording_event.is_set():
                    try:
                        # Record small chunks (0.5 seconds each)
                        audio = self.speech_recognizer.record(source, duration=0.5)
                        self.recording_audio_data.append(audio)
                    except Exception as e:
                        print(f"Recording error: {e}")
                        break
                        
        except Exception as e:
            print(f"Microphone error in recording thread: {e}")
        finally:
            print(f"Recording thread finished. Captured {len(self.recording_audio_data)} chunks")
                
    def stop_and_transcribe(self):
        """Stop recording and transcribe the audio (push-to-talk release)"""
        with self.recording_lock:
            if not self.is_recording:
                return {"success": False, "error": "Not recording"}
            
            # Signal stop
            self.is_recording = False
            self.stop_recording_event.set()
            
            # Wait for recording thread to finish (give it time to exit gracefully)
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=2.0)
        
        print(f"Recording stopped. Processing {len(self.recording_audio_data)} audio chunks...")
        
        if not self.recording_audio_data:
            return {"success": False, "error": "No audio recorded"}
        
        try:
            # Combine all audio chunks
            combined_audio = self._combine_audio_chunks(self.recording_audio_data)
            
            if combined_audio is None:
                return {"success": False, "error": "Failed to combine audio chunks"}
            
            # Transcribe using Google
            caption = self.speech_recognizer.recognize_google(
                combined_audio,
                language='en-US',
                show_all=False
            )
            
            print(f"Transcribed: {caption}")
            
            # Clear recording data
            self.recording_audio_data = []
            
            return {
                "success": True, 
                "caption": caption
            }
            
        except sr.UnknownValueError:
            return {"success": False, "error": "Could not understand audio"}
        except sr.RequestError as e:
            return {"success": False, "error": f"Recognition service error: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
        
    def _combine_audio_chunks(self, audio_chunks):
        """Combine multiple audio chunks into one AudioData object"""
        if not audio_chunks:
            return None
        
        # Combine raw audio data
        combined_frames = b''.join(chunk.frame_data for chunk in audio_chunks)
        
        # Create new AudioData object
        return sr.AudioData(
            combined_frames, 
            audio_chunks[0].sample_rate, 
            audio_chunks[0].sample_width
        )
    
    def transcribe_audio_file(self, audio_file_path):
        """Transcribe a pre-recorded audio file for testing"""
        try:
            with sr.AudioFile(audio_file_path) as source:
                audio = self.speech_recognizer.record(source)
            
            caption = self.speech_recognizer.recognize_google(
                audio,
                language='en-US'
            )
            
            return {"success": True, "caption": caption}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
        
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
    
    def run(self):
        print("Server running. Available request types:")
        print("  - 'inference': Process image for object detection")
        print("  - 'start_recording': Start push-to-talk recording")
        print("  - 'stop_and_transcribe': Stop recording and get transcription")
        print("  - 'transcribe_file': Transcribe audio file for testing")
        
        while True:
            try:
                message = self.socket.recv()
                request = json.loads(message)
                request_type = request.get("type")
                
                if request_type == "inference":
                    # Decode base64 image
                    image_data = base64.b64decode(request["image"])
                    detections = self.process_image(image_data)
                    
                    # Send response
                    response = json.dumps({
                        "success": True,
                        "detections": detections
                    })
                    self.socket.send(response.encode())

                elif request_type == "start_recording":
                    result = self.start_recording()
                    response = json.dumps(result)
                    self.socket.send(response.encode())

                elif request_type == "stop_and_transcribe":
                    result = self.stop_and_transcribe()
                    response = json.dumps(result)
                    self.socket.send(response.encode())

                elif request_type == "transcribe_file":
                    # For testing with audio files
                    audio_data = base64.b64decode(request['audio_file'])
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        tmp_file.write(audio_data)
                        tmp_path = tmp_file.name
                    
                    result = self.transcribe_audio_file(tmp_path)
                    
                    # Clean up
                    os.unlink(tmp_path)
                    
                    response = json.dumps(result)
                    self.socket.send(response.encode())
                
                else:
                    response = json.dumps({'success': False, 'error': f'Unknown type: {request_type}'})
                    self.socket.send(response.encode())
                    
            except Exception as e:
                print(f"Error in main loop: {e}")
                error_response = json.dumps({
                    'success': False,
                    'error': str(e)
                })
                self.socket.send(error_response.encode())

    def run_mock(self):
        instrument = random.choice(list(test_commands.keys()))
        caption = random.choice(test_commands[instrument])
        
        return {
            "success": True, 
            "caption": caption
        }

if __name__ == '__main__':
    model_path = "/home/pwnwas/Personal/College/Skripsi/Code/playground_ws/src/panda_vision/models/yolo26n_736_150_621i.pt"
    server = YOLOServer(model_path)

    if mock:
        server.run_mock()
    else:  
      server.run()