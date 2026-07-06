# Before You Forget

## Setup

1. Install ROS2 Humble on `Python 3.10.12`

Follow the instruction in this documentation: `https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html`

2. Clone this repository

```bash
git clone git@github.com:sulaimanfawwazak/undergrad-thesis.git
```

3. Make a ROS2 workspace

```bash
mkdir new_ws
cd new_ws
mkdir src
colcon build
source ./install/setup.bash
```

4. Move/copy the GitHub repository content to the `new_ws/src`

```bash
mv undergrad-thesis/ new_ws/src
cd new_ws
colcon build
source ./install/setup.bash
```

5. Install Miniconda

Follow the steps in this documentation: `https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install`

6. Create a new Miniconda environment (e.g `yolo`) in Python `3.11.15`

```bash
conda create -n yolo python=3.11.15
```

8. Activate the Miniconda environment

```bash
conda activate yolo
```

7. Go to `new_ws/src/outside` and install the dependencies inside Miniconda environment

```bash
cd new_ws/src/outside
pip install -r requirements.txt
```

## Run Sequence

1. Run 6-8 Terminal instances

2. Go to the 1st Terminal, go to `new_ws/src/outside`, activate the Miniconda environment, and run the YOLO OBB server

```bash
cd new_ws/src/outside
conda activate yolo
python yolo_server.py
```

3. Go to the 2nd Terminal, go to `new_ws/src/outside`, activate the Miniconda environment, and run the STT server

```bash
cd new_ws/src/outside
conda activate yolo
python stt_server.py
```

4. Go to the 3rd Terminal, go to `new_ws/`, launch the Gazebo simulation

```bash
cd new_ws
source ./install/setup.bash

# Randomize instruments
ros2 launch panda_desription panda_with_instruments.launch.py randomize_instruments:=true random_seed:=100

# Instruments in the sorted positions
ros2 launch panda_desription panda_with_instruments.launch.py randomize_instruments:=false
```

5. Go to the 4th Terminal, go to `new_ws/`, run the `yolo_detector` node

```bash
cd new_ws
source ./install/setup.bash

ros2 run panda_vision yolo_detector
```

6. Go to the 5th Terminal, go to `new_ws/`, run the `llm_controller` node

```bash
cd new_ws
source ./install/setup.bash

ros2 run panda_vision llm_controller
```

7. Go to the 6th Terminal, go to `new_ws/`, run the `instrument_orchestrator` node

```bash
cd new_ws
source ./install/setup.bash

ros2 run panda_vision instrument_orchestrator
```

8. Go to the 7th Terminal, go to `new_ws/`, run the `instrument_pick_and_place` node

```bash
cd new_ws
source ./install/setup.bash

ros2 run pymoveit2 instrument_pick_and_place.py
```

9. Go to the 8th Terminal, go to `new_ws/`, run the `speech_listener` node or publish a message to the `/speech/text` topic

```bash
cd new_ws
source ./install/setup.bash

ros2 run panda_vision speech_listener

# OR

ros2 topic pub /speech/text std_msgs/String "data: 'please pick the scalpel'" --once
```




