#!/usr/bin/env python3
"""
Test script for speech pipeline without ROS
"""

import json
import zmq
import time
import sys

def test_stt_server(port=5556, num_requests=10):
    """Test STT server connection"""
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{port}")
    
    print(f"Connected to STT server on port {port}")
    print(f"Will make {num_requests} transcription requests\n")
    
    for i in range(num_requests):
        try:
            # Send transcription request
            request = json.dumps({"type": "transcribe"})
            socket.send(request.encode())
            
            # Wait for response (with timeout)
            socket.setsockopt(zmq.RCVTIMEO, 3000)
            response = socket.recv()
            result = json.loads(response.decode())
            
            if result.get('success'):
                text = result.get('text', '')
                confidence = result.get('confidence', 0.0)
                instrument = result.get('instrument', '')
                mock = result.get('mock', False)
                
                mock_str = "[MOCK]" if mock else "[REAL]"
                print(f"{i+1:2d}. {mock_str} '{text}'")
                print(f"     Instrument: {instrument}, Confidence: {confidence:.2f}\n")
            else:
                print(f"{i+1:2d}. Error: {result.get('error', 'Unknown error')}\n")
                
        except zmq.Again:
            print(f"{i+1:2d}. Timeout - no response from server\n")
        except Exception as e:
            print(f"{i+1:2d}. Exception: {e}\n")
        
        time.sleep(1)  # Wait between requests
    
    socket.close()
    context.term()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5556
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    test_stt_server(port, num)