#!/home/pwnwas/miniconda3/envs/colab_mimic/bin/python
"""
Speech-to-Text Server for Surgical Robotics
Supports: Mock mode, Whisper (CPU/GPU), Real mic input, VAD
"""

import argparse
import json
import time
import random
import base64
import sys
import signal
from threading import Thread, Lock
from typing import Optional, Tuple
import numpy as np
import zmq
import sounddevice as sd
import webrtcvad
from charminal import *

# Test commands for mock mode
TEST_COMMANDS = {
    "scalpel": [
        "hand me the scalpel",
        "give me the scalpel",
        "i need to make an incision",
        "time to cut",
        "scalpel please",
        "making the incision now",
        "i'll incise here",
        "pass the blade",
        # "need to cut",
        # "starting the incision"
    ],
    "scissors": [
        "hand me the scissors",
        "give me the scissors",
        # "cut this suture",
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
        # "i need better exposure",
        "retractor please",
        # "let me retract this",
        # "need to expose the site",
        # "pull this back",
        # "hold this open",
        # "i need to see better",
        # "expose the area please"
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

class STTServer:
    def __init__(self, mode="mock", port=5556, model_size="base", 
                 device="cpu", vad_mode=1, sample_rate=16000, mic_device=None):
        """
        Args:
            mode: "mock", "whisper", or "file"
            port: ZeroMQ port
            model_size: "tiny", "base", "small", "medium", "large"
            device: "cpu" or "cuda"
            vad_mode: 0-3 (0=least aggressive, 3=most aggressive)
            sample_rate: Audio sample rate (16000 for Whisper)
        """
        self.mode = mode
        self.port = port
        self.model_size = model_size
        self.device = device
        self.vad = webrtcvad.Vad(vad_mode)
        self.sample_rate = sample_rate
        self.frame_duration_ms = 30  # VAD works on 10-30ms frames
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        self.mic_device = mic_device
        
        # Audio buffer for real-time processing
        self.audio_buffer = []
        self.buffer_lock = Lock()
        self.is_recording = False
        self.speech_detected = False
        self.silence_counter = 0
        self.silence_timeout_frames = 30  # ~1 second of silence to stop
        
        # Store latest transcription
        self.latest_transcription = None
        self.transcription_lock = Lock()
        
        # Whisper model (lazy loaded)
        self.whisper_model = None
        
        # Mock mode data
        self.mock_commands = []
        for instrument, commands in TEST_COMMANDS.items():
            for cmd in commands:
                self.mock_commands.append({
                    "text": cmd,
                    "instrument": instrument,
                    "confidence": 0.95
                })
        
        # ZeroMQ setup
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        
        # Running flag
        self.running = True
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Audio stream
        self.audio_stream = None
        
        # Audio callback thread for continuous listening
        self.audio_thread = None
        
        print(f"{COLOR_GREEN}STT Server initialized:{RESET}")
        print(f"  Mode: {mode}")
        print(f"  Port: {port}")
        print(f"  Sample rate: {sample_rate} Hz")
        if mode == "whisper":
            print(f"  Whisper model: {model_size}")
            print(f"  Device: {device}")
            print(f"  VAD mode: {vad_mode}")
        elif mode == "mock":
            print(f"  Mock commands: {len(self.mock_commands)}")
        
    def signal_handler(self, sig, frame):
        print(f"\n{COLOR_YELLOW}Shutting down STT server...{RESET}")
        self.running = False
        
    def load_whisper_model(self):
        """Lazy load Whisper model"""
        if self.whisper_model is not None:
            return
            
        try:
            from faster_whisper import WhisperModel
            
            # Determine compute type based on device
            if self.device == "cuda":
                compute_type = "float16"
            else:
                compute_type = "int8"
                
            print(f"{COLOR_CYAN}Loading Whisper model '{self.model_size}' on {self.device}...{RESET}")
            # Suppress HF warnings
            import warnings
            warnings.filterwarnings("ignore", message="You are sending unauthenticated requests")
            
            self.whisper_model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=compute_type,
                cpu_threads=4,
                num_workers=1
            )
            print(f"{COLOR_GREEN}Whisper model loaded!{RESET}")
            
        except Exception as e:
            print(f"{COLOR_RED}Failed to load Whisper model: {e}{RESET}")
            print(f"{COLOR_YELLOW}Falling back to mock mode{RESET}")
            self.mode = "mock"
    
    def transcribe_audio(self, audio_data: np.ndarray) -> Tuple[str, float]:
        """
        Transcribe audio using Whisper
        Args:
            audio_data: Float32 array of shape (samples,)
        Returns:
            (text, confidence)
        """
        if self.whisper_model is None:
            self.load_whisper_model()
            
        try:
            segments, info = self.whisper_model.transcribe(
                audio_data,
                language="en",
                beam_size=5,
                best_of=5,
                temperature=0.0,
                vad_filter=True
            )

            segments = list(segments)
            
            # Collect all segments
            text = " ".join([seg.text for seg in segments])
            
            if segments:
                # Average confidence across segments (using no_speech_prob as proxy)
                # Note: This is a heuristic - adjust based on your needs
                avg_confidence = sum(1.0 - seg.no_speech_prob for seg in segments) / len(segments)
                confidence = max(0.0, min(1.0, avg_confidence))
            else:
                confidence = 0.0
            
            # confidence = sum(seg.avg_log_prob for seg in segments) / len(segments) if segments else 0.0
            # confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
            
            return text.strip(), confidence
            
        except Exception as e:
            print(f"{COLOR_RED}Transcription error: {e}{RESET}")
            return "", 0.0
    
    def process_audio_chunk(self, chunk_bytes: bytes):
        """Process audio chunk with VAD"""
        # Check if chunk contains speech (webrtcvad expects bytes directly)
        is_speech = self.vad.is_speech(chunk_bytes, self.sample_rate)
        
        if is_speech:
            if not self.is_recording:
                # Start new recording
                self.is_recording = True
                self.audio_buffer = []
                self.speech_detected = True
                print(f"{COLOR_CYAN}Speech detected, recording...{RESET}")
            
            # Convert bytes to int16 numpy array and append
            chunk_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
            self.audio_buffer.append(chunk_int16)
            self.silence_counter = 0
            
        elif self.is_recording:
            # Silence during recording
            self.silence_counter += 1
            chunk_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
            self.audio_buffer.append(chunk_int16)
            
            if self.silence_counter > self.silence_timeout_frames:
                # End of speech
                self.is_recording = False
                print(f"{COLOR_GREEN}Speech ended, transcribing...{RESET}")
                
                # Process the recorded audio
                audio_int16 = np.concatenate(self.audio_buffer)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                
                text, confidence = self.transcribe_audio(audio_float32)
                
                if text:
                    print(f"{COLOR_GREEN}Transcribed: '{text}' (conf: {confidence:.2f}){RESET}")
                    with self.transcription_lock:
                        self.latest_transcription = (text, confidence)
                else:
                    print(f"{COLOR_YELLOW}No speech recognized{RESET}")
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice InputStream"""
        if status:
            print(f"{COLOR_YELLOW}Audio status: {status}{RESET}")
        
        # indata contains the audio data as bytes (since dtype='int16')
        # Convert to bytes for VAD
        chunk_bytes = indata.tobytes()
        self.process_audio_chunk(chunk_bytes)
    
    def start_audio_stream(self):
        """Start continuous audio stream with callback"""
        if self.audio_stream is None:
            self.audio_stream = sd.InputStream(
                device=self.mic_device,
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16',
                blocksize=self.frame_size,
                callback=self.audio_callback
            )
            self.audio_stream.start()
            print(f"{COLOR_GREEN}Audio stream started{RESET}")
    
    def get_mock_command(self) -> dict:
        """Return a random mock command"""
        cmd = random.choice(self.mock_commands)
        return {
            "text": cmd["text"],
            "confidence": cmd["confidence"],
            "instrument": cmd["instrument"],
            "mock": True
        }
    
    def run(self):
        """Main server loop with improved reliability"""
        print(f"{COLOR_GREEN}STT Server running on port {self.port}{RESET}")
        print(f"{COLOR_YELLOW}Waiting for connections...{RESET}")
        
        if self.mode == "whisper":
            self.start_audio_stream()
        
        # Track last response to avoid duplicate sends
        last_response = None
        last_response_time = 0
        
        while self.running:
            try:
                if self.socket.poll(100):
                    message = self.socket.recv()
                    request = json.loads(message.decode())
                    
                    response = None
                    
                    if request.get("type") == "transcribe":
                        if self.mode == "mock":
                            # Only send new command every few seconds in mock mode
                            current_time = time.time()
                            if current_time - last_response_time > 0.5 or last_response is None:
                                cmd = self.get_mock_command()
                                response = {
                                    "success": True,
                                    "text": cmd["text"],
                                    "confidence": cmd["confidence"],
                                    "instrument": cmd["instrument"],
                                    "mock": True
                                }
                                last_response = response
                                last_response_time = current_time
                                print(f"{COLOR_CYAN}[MOCK] Sending: '{cmd['text']}'{RESET}")
                            else:
                                # Send cached response to avoid overwhelming
                                response = last_response
                                
                        elif self.mode == "whisper":
                            with self.transcription_lock:
                                if self.latest_transcription:
                                    text, confidence = self.latest_transcription
                                    self.latest_transcription = None
                                    response = {
                                        "success": True,
                                        "text": text,
                                        "confidence": confidence,
                                        "mock": False
                                    }
                                else:
                                    response = {
                                        "success": False,
                                        "error": "No speech detected yet"
                                    }
                    
                    elif request.get("type") == "ping":
                        response = {"success": True, "status": "alive", "mode": self.mode}
                    
                    else:
                        response = {"success": False, "error": f"Unknown request type"}
                    
                    if response:
                        self.socket.send(json.dumps(response).encode())
                
                time.sleep(0.01)
                    
            except zmq.ZMQError as e:
                print(f"{COLOR_RED}ZMQ Error: {e}{RESET}")
                # Don't exit, just continue
                time.sleep(0.1)
            except Exception as e:
                print(f"{COLOR_RED}Unexpected error: {e}{RESET}")
                time.sleep(0.1)
        
        # Cleanup
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
        self.socket.close()
        self.zmq_context.term()
        print(f"{COLOR_GREEN}STT Server shutdown complete{RESET}")


def list_audio_devices():
    """List available audio input devices"""
    print(f"{COLOR_CYAN}Available audio input devices:{RESET}")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  {i}: {device['name']} (channels: {device['max_input_channels']})")


def main():
    parser = argparse.ArgumentParser(description="STT Server for Surgical Robotics")
    parser.add_argument("--mode", type=str, default="mock",
                        choices=["mock", "whisper", "file"],
                        help="Operation mode: mock, whisper, or file")
    parser.add_argument("--port", type=int, default=5556,
                        help="ZeroMQ port (default: 5556)")
    parser.add_argument("--model", type=str, default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Compute device for Whisper (default: cpu)")
    parser.add_argument("--vad-mode", type=int, default=1, choices=[0, 1, 2, 3],
                        help="VAD aggressiveness (0=least, 3=most)")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Audio sample rate (default: 16000)")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio devices and exit")
    parser.add_argument("--audio-file", type=str,
                        help="Audio file to process (file mode only)")
    parser.add_argument("--mic-device", type=int, default=None, help="Input microphone index")
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        return
    
    if args.mode == "file" and not args.audio_file:
        print(f"{COLOR_RED}Error: --audio-file required for file mode{RESET}")
        return
    
    if args.mic_device is None:
        args.mic_device = sd.default.device[0]
    
    server = STTServer(
        mode=args.mode,
        port=args.port,
        model_size=args.model,
        device=args.device,
        vad_mode=args.vad_mode,
        sample_rate=args.sample_rate,
        mic_device=args.mic_device
    )
    
    # Load Whisper model if needed
    if args.mode == "whisper":
        server.load_whisper_model()
        if args.audio_file:
            # Process single file and exit
            result = server.process_file(args.audio_file)
            print(f"Result: {json.dumps(result, indent=2)}")
            return
    
    try:
        server.run()
    except KeyboardInterrupt:
        print(f"\n{COLOR_YELLOW}Interrupted by user{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()