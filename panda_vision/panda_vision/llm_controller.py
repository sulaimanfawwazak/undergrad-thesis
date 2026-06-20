#!/usr/bin/env python3
"""
LLM Controller for Surgical Robot
Decides which instrument to pick based on speech command and available instruments
Supports: Mock mode (rule-based) and Gemini API mode
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import argparse
from typing import List, Dict, Optional, Literal, Tuple
from enum import Enum
import random
import os
from charminal import *
import numpy as np
import math

# Pydantic for structured output (optional for mock mode, required for Gemini)
try:
    from pydantic import BaseModel, Field
    from typing import Optional, Literal
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print(f"{COLOR_YELLOW}Warning: pydantic not installed. Install with: pip install pydantic{RESET}")

# Gemini API (optional)
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print(f"{COLOR_YELLOW}Warning: google-genai not installed. Gemini mode will not work.{RESET}")
    print(f"Install with: pip install google-genai")

# Define instrument list
INSTRUMENTS = ["needle_holder", "scalpel", "scissors", "tweezers", "retractor"]
INSTRUMENT_ALIASES = {
    "needle_holder": ["needle holder", "needle driver", "suture"],
    "scalpel": ["blade", "incision", "cut", "knife"],
    "scissors": ["scissor", "snip", "cut suture", "dissect"],
    "tweezers": ["forceps", "grasp", "pick up"],
    "retractor": ["retract", "expose", "pull back", "hold open"]
}

# Pydantic model for structured output
class RobotCommand(BaseModel):
    """Robot command structure for LLM output"""
    operation: Literal["pick", "sort", "pick_sorted", "none"] = Field(
        description="Operation to perform: pick (from current table), sort (organize instruments), pick_sorted (from sorted table), none (no action)"
    )
    instrument: Optional[Literal["needle_holder", "scalpel", "scissors", "tweezers", "retractor"]] = Field(
        default=None,
        description="Instrument to pick (required for pick operations)"
    )
    confidence: float = Field(
        description="Confidence in this decision (0.0 to 1.0)",
        ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of why this decision was made"
    )
    source_table: Optional[Literal["table1", "table2"]] = Field(
        default="table1",
        description="Which table to pick from"
    )
    # dest_table: Literal["table1", "table2"] = Field(
    #     default="table2",
    #     description="Which table to place to (for sort operation)"
    # )


class MockLLM:
    """Rule-based mock LLM for testing without API calls"""
    
    def __init__(self):
        self.instruments = INSTRUMENTS
        self.sort_keywords = ["sort", "organize", "arrange", "clean up", "put away", "arrange"]
        
    def extract_instrument(self, text_lower: str) -> Optional[str]:
        """Extract instrument name from text using keywords"""
        for instrument in self.instruments:
            # Direct match
            if instrument.replace('_', ' ') in text_lower:
                return instrument
            # Check aliases
            for alias in INSTRUMENT_ALIASES.get(instrument, []):
                if alias in text_lower:
                    return instrument
        return None
    
    def decide(self, speech_text: str, available_instruments: List[str], 
               instruments_sorted: bool) -> Dict:
        """Make decision based on rules"""
        text_lower = speech_text.lower()
        
        # Handle empty/no instruments case
        if not available_instruments:
            return {
                "operation": "none",
                "instrument": None,
                "confidence": 0.0,
                "reasoning": "No instruments detected on table",
                "source_table": "table1",
                "dest_table": "table2"
            }
        
        # Check for sort command
        if any(kw in text_lower for kw in self.sort_keywords):
            if not instruments_sorted:
                return {
                    "operation": "sort",
                    "instrument": None,
                    "confidence": 0.95,
                    "reasoning": f"User requested sorting instruments",
                    "source_table": "table1",
                    # "dest_table": "table2"
                }
            else:
                return {
                    "operation": "none",
                    "instrument": None,
                    "confidence": 0.7,
                    "reasoning": "Instruments already sorted, nothing to sort",
                    # "source_table": "table2",
                    "source_table": None,
                    # "dest_table": "table2"
                }
        
        # Check for specific instrument
        instrument = self.extract_instrument(text_lower)
        
        if instrument and instrument in available_instruments:
            if instruments_sorted:
                operation = "pick_sorted"
                source_table = "table2"
                reasoning = f"User asked for {instrument} (instruments are on sorted table)"
            else:
                operation = "pick"
                source_table = "table1"
                reasoning = f"User asked for {instrument} (instruments on unsorted table)"
            
            return {
                "operation": operation,
                "instrument": instrument,
                "confidence": 0.95,
                "reasoning": reasoning,
                "source_table": source_table,
                # "dest_table": "table2"
            }
        
        # Instrument not available
        if instrument and instrument not in available_instruments:
            return {
                "operation": "none",
                "instrument": None,
                "confidence": 0.5,
                "reasoning": f"User asked for {instrument} but it's not available. Available: {', '.join(available_instruments)}",
                # "source_table": "table1",
                "source_table": None,
                # "dest_table": "table2"
            }
        
        # No match
        return {
            "operation": "none",
            "instrument": None,
            "confidence": 0.3,
            "reasoning": f"Could not understand command: '{speech_text}'",
            # "source_table": "table1",
            "source_table": None,
            # "dest_table": "table2"
        }


class GeminiLLM:
    """Gemini API-based LLM for intelligent decision making"""
    
    def __init__(self, api_key: str, model: str = "gemini-3.1-flash-lite"):
        self.api_key = api_key  # Don't hardcode! Use parameter
        self.model = model
        self.client = genai.Client(api_key=api_key)
        
        if not PYDANTIC_AVAILABLE:
            raise ImportError("pydantic is required for Gemini mode. Install with: pip install pydantic")
        
        print(f"{COLOR_GREEN}Gemini LLM initialized with model: {model}{RESET}")
    
    def format_instruments_list(self, instruments: List[str], positions: List[List[float]], rotations: List[float]) -> str:
        """Format instruments with their positions and rotations for the prompt"""
        formatted = []
        for i, (inst, pos, rot) in enumerate(zip(instruments, positions, rotations), 1):
            # Format the 8 corner points nicely
            # corners = pos[:8]  # xyxyxyxy format
            # corner_str = f"({corners[0]:.1f}, {corners[1]:.1f}, {corners[2]:.1f}, {corners[3]:.1f}, {corners[4]:.1f}, {corners[5]:.1f}, {corners[6]:.1f}, {corners[7]:.1f})"

            # Format the centers (cx, cy) nicely
            center_str = f"({pos[0]}, {pos[1]})"
            rot_deg = math.degrees(rot) # Convert radians to degrees for readability
            
            # formatted.append(f"{i}. {inst}: corners={corner_str}, rotation={rot_deg:.1f}° ({rot:.3f} rad)")
            formatted.append(f"{i}. {inst}: center={center_str}, rotation={rot_deg:.1f}° ({rot:.3f} rad)")

        return '\n'.join(formatted)

    def build_prompt(self, speech_text: str, available_instruments: List[str], 
                     instruments_positions: List[List[float]], 
                     instruments_rotations: List[float],
                     instruments_sorted: bool, current_table: str) -> str:
        """Build prompt for Gemini API with instrument positions and rotations"""
        
        instruments_str = self.format_instruments_list(available_instruments, instruments_positions, instruments_rotations)
        
        prompt = f"""
You are an AI assistant for a surgical robot. You can understand spatial context from a down-looking camera and a YOLO detector to recognize surgical instruments available on the table.
The robot has two tables:
- Table 1: Table for initial position for the instruments, where coordinate and rotation of each instruments are randomized
- Table 2: Table where the instruments are sorted/organized/arranged (after surgeon asked for sorting)

Current state:
- Instruments available on table, with their positions and orientations:
{instruments_str}

- The MOST-LEFT instrument has the LOWEST X value
- The MOST-RIGHT instrument has the HIGHEST X value
- The MOST-TOP instrument has the LOWEST Y value
- The MOST-BOTTOM instrument has the HIGHEST Y value

- Are instruments sorted: {instruments_sorted} (True = already on Table 2, False = on Table 1)
- Current instrument location: {current_table}

Available operations:
1. "pick" - Pick an instrument from current table (Table 1 IF unsorted, Table 2 IF sorted)
2. "sort" - Move all instruments from Table 1 to Table 2 in organized manner
3. "pick_sorted" - Pick an instrument from Table 2 (only valid if sorted=True)
4. "none" - No action

Rules:
- If surgeon asks for a specific instrument (scalpel, scissors, tweezers, retractor, needle_holder):
  * If instruments_sorted=False: use "pick" operation from Table 1
  * If instruments_sorted=True: use "pick_sorted" operation from Table 2
- If surgeon says "sort", "organize", "arrange" the instruments: use "sort" operation
- If surgeon asks for an instrument not available: respond with "none"
- For vague instrument commands:
  * "make an incision", "cut", "blade" → infer "scalpel"
  * "cut suture", "snip" → infer "scissors"
  * "grasp tissue", "forceps" → infer "tweezers"
  * "retract", "expose" → infer "retractor"
  * "suture", "stitch", "needle" → infer "needle_holder"
- For vague spatial commands:
  * "Grab the most-left instrument" -> Determine which instrument is the most-left, using the given available instrument coordinate above
  * "Pick the one on the right of scissors" -> Identify which instrument is on the right of scissors, on the same row, using the given available instrument coordinate above
  * "Give me the instrument at the bottom-left" -> Decide which instrument is at the bottom-left, using the given available instrument coordinate above
- When confidence is low (uncertain), set confidence accordingly (0.3-0.6)
- Provide brief reasoning for your decision

Surgeon said: "{speech_text}"

Give an appropriate respond to the surgeon's input!

Respond ONLY with a JSON object matching this exact schema (no additional text):
{{
    "operation": "pick" or "sort" or "pick_sorted" or "none",
    "instrument": "instrument_name" or null,
    "confidence": 0.95,
    "reasoning": "brief explanation",
    "source_table": "table1" or "table2" or null
}}
"""
        print(f"{COLOR_MAGENTA}Prompt: {prompt}{RESET}")

        return prompt
    
    def decide(self, speech_text: str, available_instruments: List[str],
               instruments_positions: List[List[float]],
               instruments_rotations: List[float],
               instruments_sorted: bool) -> Dict:
        """Call Gemini API to make decision"""
        
        current_table = "table2" if instruments_sorted else "table1"
        
        prompt = self.build_prompt(
            speech_text=speech_text,
            available_instruments=available_instruments,
            instruments_positions=instruments_positions,
            instruments_rotations=instruments_rotations,
            instruments_sorted=instruments_sorted,
            current_table=current_table
        )
        
        try:
            # Call Gemini API
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.2,  # Low temperature for consistent outputs
                }
            )

#             response = '''
#                 {
#                     "operation": "none",
#                     "instrument": null,
#                     "confidence": 0.95,
#                     "reasoning": "nothing",
#                     "source_table": null
#                 }
# '''
            
            # Parse JSON response
            result = json.loads(response.text)
            
            # Validate against Pydantic model
            command = RobotCommand(**result)
            
            # Return as dict
            return command.model_dump()
            
        except Exception as e:
            print(f"{COLOR_RED}Gemini API error: {e}{RESET}")
            # Fallback to safe response
            return {
                "operation": "none",
                "instrument": None,
                "confidence": 0.0,
                "reasoning": f"API error: {str(e)}",
                # "source_table": current_table,
                "source_table": None,
                # "dest_table": "table2"
            }


class LLMController(Node):
    """ROS2 Node that controls LLM decision making for surgical robot"""
    
    def __init__(self, mode="mock", api_key=None, model="gemini-3.1-flash-lite"):
        super().__init__('llm_controller')
        
        self.mode = mode
        self.instruments_sorted = False
        self.current_table = "table1"
        
        # Store instrument data with their positions and rotations
        self.latest_instruments = []      # List of instrument names
        self.latest_positions = []        # List of corner coordinates (8 points each)
        self.latest_rotations = []        # List of rotation angles (radians)
        self.latest_speech = None
        
        # Publishers
        self.command_pub = self.create_publisher(String, '/robot/command', 10)
        self.debug_pub = self.create_publisher(String, '/llm/debug', 10)
        
        # Subscribers
        self.speech_sub = self.create_subscription(
            String,
            '/speech/text',
            self.speech_callback,
            10
        )
        self.yolo_sub = self.create_subscription(
            String,
            '/yolo/detection',
            self.yolo_callback,
            10
        )
        
        # Initialize LLM based on mode
        if mode == "mock":
            self.llm = MockLLM()
            self.get_logger().info(f"{COLOR_GREEN}LLM Controller started in MOCK mode{RESET}")
        elif mode == "gemini":
            if not GEMINI_AVAILABLE:
                self.get_logger().error("Gemini mode requested but google-genai not installed")
                raise ImportError("Install google-genai: pip install google-genai")
            if not api_key:
                self.get_logger().error("Gemini mode requested but no API key provided")
                raise ValueError("API key required for Gemini mode")
            self.llm = GeminiLLM(api_key=api_key, model=model)
            self.get_logger().info(f"{COLOR_GREEN}LLM Controller started in GEMINI mode (model: {model}){RESET}")
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Timer for periodic status (optional)
        self.status_timer = self.create_timer(10.0, self.publish_status)
        
        self.get_logger().info(f"{COLOR_CYAN}LLM Controller Node Ready{RESET}")
        self.get_logger().info(f"  Mode: {mode}")
        self.get_logger().info(f"  Sorted: {self.instruments_sorted}")
    
    def yolo_callback(self, msg: String):
        """Process YOLO detections and update available instruments with positions and rotations"""
        try:
            detections = json.loads(msg.data)
            
            # Clear previous data
            self.latest_instruments = []
            self.latest_positions = []
            self.latest_rotations = []
            
            # Process each detection
            for det in detections:
                det_x1, det_y1, det_x2, det_y2, det_x3, det_y3, det_x4, det_y4 = det["xyxyxyxy"]
                center = [det["xywhr"][0], det["xywhr"][1]]

                y_min = min(det_y1, det_y2, det_y3, det_y4)

                if y_min == det_y1:
                    # Find distance between x1-x2 and x1-x4
                    dist_1 = np.sqrt((det_x1 - det_x2)**2 + (det_y1 - det_y2)**2)
                    dist_2 = np.sqrt((det_x1 - det_x4)**2 + (det_y1 - det_y4)**2)

                    if dist_1 > dist_2:
                        dx = det_x2 - det_x1
                        dy = det_y2 - det_y1
                        angle = np.arctan2(dy, dx)
                    else:
                        dx = det_x4 - det_x1
                        dy = det_y4 - det_y1
                        angle = np.arctan2(dy, dx)

                elif y_min == det_y2:
                    # Find distance between x2-x1 and x2-x3
                    dist_1 = np.sqrt((det_x2 - det_x1)**2 + (det_y2 - det_y1)**2)
                    dist_2 = np.sqrt((det_x2 - det_x3)**2 + (det_y2 - det_y3)**2)

                    if dist_1 > dist_2:
                        dx = det_x1 - det_x2
                        dy = det_y1 - det_y2
                        angle = np.arctan2(dy, dx)
                    else:
                        dx = det_x3 - det_x2
                        dy = det_y3 - det_y2
                        angle = np.arctan2(dy, dx)

                elif y_min == det_y3:
                    # Find distance between x3-x2 and x3-x4
                    dist_1 = np.sqrt((det_x3 - det_x2)**2 + (det_y3 - det_y2)**2)
                    dist_2 = np.sqrt((det_x3 - det_x4)**2 + (det_y3 - det_y4)**2)

                    if dist_1 > dist_2:
                        dx = det_x2 - det_x3
                        dy = det_y2 - det_y3
                        angle = np.arctan2(dy, dx)
                    else:
                        dx = det_x4 - det_x3
                        dy = det_y4 - det_y3
                        angle = np.arctan2(dy, dx)

                elif y_min == det_y4:
                    # Find distance between x4-x1 and x4-x3
                    dist_1 = np.sqrt((det_x4 - det_x1)**2 + (det_y4 - det_y1)**2)
                    dist_2 = np.sqrt((det_x4 - det_x3)**2 + (det_y4 - det_y3)**2)

                    if dist_1 > dist_2:
                        dx = det_x1 - det_x4
                        dy = det_y1 - det_y4
                        angle = np.arctan2(dy, dx)
                    else:
                        dx = det_x3 - det_x4
                        dy = det_y3 - det_y4
                        angle = np.arctan2(dy, dx)

                angle = -1 * angle

                instrument = det['class_name']
                corners = det['xyxyxyxy']  # List of 8 numbers (x1,y1,x2,y2,x3,y3,x4,y4)
                center = det['xywhr']
                rotation = angle 

                print(f"{COLOR_CYAN}Instrument: {instrument}, coordinate: ({center[0], center[1]}) angle: {angle} rad ({math.degrees(angle)}°){RESET}")
                
                self.latest_instruments.append(instrument)
                # self.latest_positions.append(corners)
                self.latest_positions.append(center)
                self.latest_rotations.append(rotation)
            
            # Debug output
            if self.latest_instruments:
                self.get_logger().debug(f"YOLO detected: {list(zip(self.latest_instruments, self.latest_rotations))}")
            
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse YOLO detection: {e}")
        except Exception as e:
            self.get_logger().error(f"Error processing YOLO callback: {e}")
    
    def publish_command(self, command: Dict):
        """Publish robot command to topic"""
        msg = String()
        msg.data = json.dumps(command)
        self.command_pub.publish(msg)
        
        # Log the command
        if command['operation'] != 'none':
            self.get_logger().info(f"{COLOR_GREEN}Command: {command['operation']} {command.get('instrument', '')} (conf: {command['confidence']:.2f}){RESET}")
        else:
            self.get_logger().info(f"{COLOR_YELLOW}Command: none - {command['reasoning']}{RESET}")
        
        # Update state based on command
        if command['operation'] == 'sort' and command['confidence'] > 0.7:
            self.instruments_sorted = True
            self.current_table = "table2"
            self.get_logger().info(f"{COLOR_CYAN}State updated: instruments_sorted=True, current_table=table2{RESET}")
    
    def publish_debug(self, data: Dict):
        """Publish debug information"""
        msg = String()
        msg.data = json.dumps(data)
        self.debug_pub.publish(msg)
    
    def publish_status(self):
        """Publish current status periodically"""
        status = {
            "mode": self.mode,
            "instruments_sorted": self.instruments_sorted,
            "current_table": self.current_table,
            "available_instruments": self.latest_instruments,
            "instruments_rotations_deg": [r * 180 / 3.14159 for r in self.latest_rotations] if self.latest_rotations else [],
            "last_speech": self.latest_speech
        }
        self.publish_debug(status)
    
    def speech_callback(self, msg: String):
        """Process speech command and make decision"""
        speech_text = msg.data
        self.latest_speech = speech_text
        
        self.get_logger().info(f"{COLOR_CYAN}Received speech: '{speech_text}'{RESET}")
        
        # Check if we have YOLO detections
        if not self.latest_instruments:
            self.get_logger().warn("No instruments detected by YOLO")
            # Still try to process (LLM will handle empty list)
        
        # Make decision using LLM
        if self.mode == "mock":
            decision = self.llm.decide(
                speech_text=speech_text,
                available_instruments=self.latest_instruments,
                instruments_sorted=self.instruments_sorted
            )
        else:  # gemini mode
            decision = self.llm.decide(
                speech_text=speech_text,
                available_instruments=self.latest_instruments,
                instruments_positions=self.latest_positions,
                instruments_rotations=self.latest_rotations,
                instruments_sorted=self.instruments_sorted
            )
        
        # Publish the command
        self.publish_command(decision)
        
        # Publish debug info
        debug_info = {
            "speech": speech_text,
            "available_instruments": self.latest_instruments,
            "instruments_rotations_rad": self.latest_rotations,
            "instruments_sorted": self.instruments_sorted,
            "decision": decision
        }
        self.publish_debug(debug_info)
    
    def destroy_node(self):
        self.get_logger().info("Shutting down LLM Controller")
        super().destroy_node()


def main(args=None):
    # Parse command line arguments
    import sys
    
    # Simple argument parsing (since ROS2 doesn't handle these well)
    mode = "mock"
    api_key = None
    model = "gemini-3.1-flash-lite"
    
    # Parse sys.argv
    for i, arg in enumerate(sys.argv):
        if arg == "--mode" and i+1 < len(sys.argv):
            mode = sys.argv[i+1]
        elif arg == "--api-key" and i+1 < len(sys.argv):
            api_key = sys.argv[i+1]
        elif arg == "--model" and i+1 < len(sys.argv):
            model = sys.argv[i+1]
    
    # If no API key provided via args, try environment variable
    if mode == "gemini" and not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", None)
        if not api_key:
            print(f"{COLOR_RED}Error: Gemini mode requires API key. Set GEMINI_API_KEY environment variable or pass --api-key{RESET}")
            return
    
    rclpy.init(args=args)
    
    try:
        node = LLMController(mode=mode, api_key=api_key, model=model)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"{COLOR_RED}Fatal error: {e}{RESET}")
        import traceback
        traceback.print_exc()
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()