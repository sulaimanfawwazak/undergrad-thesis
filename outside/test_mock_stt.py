#!/home/pwnwas/miniconda3/envs/colab_mimic/bin/python
"""
Quick test to verify mock STT server works
Run this after starting the STT server
"""

import json
import zmq
import time

def test_mock():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5556")
    
    print("Testing mock STT server...\n")
    
    for i in range(5):
        socket.send(json.dumps({"type": "transcribe"}).encode())
        response = json.loads(socket.recv().decode())
        
        if response['success']:
            print(f"Command {i+1}: '{response['text']}' -> {response['instrument']}")
        else:
            print(f"Error: {response.get('error')}")
        
        time.sleep(1)
    
    socket.close()
    context.term()

if __name__ == "__main__":
    test_mock()