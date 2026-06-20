#!/home/pwnwas/miniconda3/envs/colab_mimic/bin/python
"""
Record test audio files for different surgical commands
Usage: python record_test_audio.py
"""

import speech_recognition as sr
import wave
import os
import time

class TestAudioRecorder:
    def __init__(self, output_dir="test_audio"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Calibrate
        with self.microphone as source:
            print("Calibrating...")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        
        self.commands = [
            "give me the scalpel",
            "hand me the scissors", 
            "i need to make an incision",
            "I want to cut this",
            "pass the needle holder",
            "retractor please",
            "tweezers"
        ]
        
        self.speakers = ["person1", "person2", "person3"]  # For robustness testing
    
    def record_command(self, command, speaker="default", take=1):
        """Record a single command"""
        print(f"\n📢 Say: '{command}'")
        print("Press Enter when ready...")
        input()
        
        print("🎙️ Recording... (3 seconds)")
        with self.microphone as source:
            try:
                audio = self.recognizer.record(source, duration=3)
                
                # Save to file
                filename = f"{self.output_dir}/{speaker}_{command[:20]}_{take}.wav"
                with open(filename, "wb") as f:
                    f.write(audio.get_wav_data())
                
                print(f"✅ Saved to: {filename}")
                
                # Test transcription immediately
                try:
                    text = self.recognizer.recognize_google(audio)
                    print(f"📝 Recognized as: '{text}'")
                except:
                    print("⚠️ Could not recognize")
                
                return filename
                
            except Exception as e:
                print(f"❌ Error: {e}")
                return None
    
    def record_test_suite(self):
        """Record multiple variations for testing"""
        print("🎤 Test Audio Recorder")
        print("=====================")
        print("This will record multiple commands for testing")
        print("You can have different people say the same commands\n")
        
        for command in self.commands:
            for speaker in self.speakers:
                for take in range(2):  # 2 takes per speaker per command
                    self.record_command(command, speaker, take+1)
                    time.sleep(1)
        
        print("\n✅ All test recordings complete!")

if __name__ == "__main__":
    recorder = TestAudioRecorder()
    recorder.record_test_suite()