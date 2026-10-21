# ARES UR5e + Robotiq MuJoCo Data Collection --- Linux Setup

This guide sets up `ares_record.py` on a new **Ubuntu/Linux PC**. It is
based on the uploaded script and its expected MuJoCo model names.

> **Important:** The Python script does not include the robot model.
> Copy `vla_scene.xml` and every XML, mesh, texture, and other asset it
> references, keeping the relative folder structure intact.

## 1. Requirements

-   64-bit Ubuntu/Linux desktop with working graphics/OpenGL
-   **Python 3.12.x** (the known working setup used Python 3.12.9)
-   Git
-   MuJoCo scene and all referenced assets
-   A supported game controller
-   Internet access for installing packages

The script uses `mujoco`, `pygame`, `numpy`, and
`lerobot.datasets.LeRobotDataset`, plus Python standard-library modules.

## 2. Check your Linux and Python versions

Open Terminal:

``` bash
cat /etc/os-release
python3 --version
git --version
```

If `python3 --version` shows Python 3.12.x, continue to Section 3.

### Install Python 3.12 on Ubuntu

On Ubuntu 24.04, Python 3.12 is normally available from the standard
repositories:

``` bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev git build-essential
```

If those packages are unavailable on your Ubuntu release, check the
official Ubuntu/Python packaging guidance for your release before adding
a third-party repository. Do not replace or remove Ubuntu's system
Python.

Verify:

``` bash
python3.12 --version
```

It should print `Python 3.12.x`.

## 3. Install system dependencies

``` bash
sudo apt update
sudo apt install git python3.12-venv python3.12-dev build-essential libgl1 libgl1-mesa-dri libglib2.0-0
```

These packages provide Git, Python environment/build support, and common
graphics/runtime libraries. Your graphics driver must also support
OpenGL.

## 4. Put the project files in one folder

For example:

``` text
/home/mars/Downloads/mjprojects/
├── ares_record.py
└── universal_robots_ur5e/
    ├── vla_scene.xml
    └── ...all XML/mesh assets referenced by the scene...
```

Use your actual folder layout; the example is not mandatory. Do not copy
only the XML file. Preserve the relative locations of all referenced
assets.

### Set `MODEL_PATH`

Open `ares_record.py` and set `MODEL_PATH` to the actual Linux path of
`vla_scene.xml`. For example, if the XML is at the location shown above:

``` python
MODEL_PATH = r"/home/mars/Downloads/mjprojects/universal_robots_ur5e/vla_scene.xml"
```

Linux paths are case-sensitive. Confirm the file exists:

``` bash
ls -l "/home/mars/Downloads/mjprojects/universal_robots_ur5e/vla_scene.xml"
```

### Set `DATASET_ROOT`

Choose a writable folder for the dataset. For example:

``` python
DATASET_ROOT = Path(r"/home/mars/lerobot/datasets/ares_ur5e_pick_cube")
```

The directory can be elsewhere, but use the same path on later runs if
you want the script to resume the existing dataset.

## 5. Create and activate a virtual environment

Choose the folder where you want to keep the script. Example:

``` bash
cd ~/Downloads/mjprojects
python3.12 -m venv .venv
source .venv/bin/activate
```

If your script is in a different directory, `cd` to that directory
instead. After activation, the prompt should begin with `(.venv)`.

Upgrade pip tools:

``` bash
python -m pip install --upgrade pip setuptools wheel
```

**Important:** use `python3.12 -m venv .venv` on Linux, not the Windows
command `py -3.12 -m venv .venv`.

## 6. Install LeRobot and required Python libraries

The recording script imports `LeRobotDataset`, so LeRobot must be
installed in the **same virtual environment** used to run the script.

### Option A --- Install the official LeRobot repository

If LeRobot is not already available on this PC:

``` bash
cd ~
git clone https://github.com/huggingface/lerobot.git
cd ~/lerobot
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install mujoco pygame numpy
```

Keep `ares_record.py` and your MuJoCo model/assets in their own project
folder if you prefer, but activate `~/lerobot/.venv` before running the
script. For example:

``` bash
source ~/lerobot/.venv/bin/activate
cd ~/Downloads/mjprojects
python ares_record.py
```

If the repository's current installation instructions differ, follow the
official guide: <https://github.com/huggingface/lerobot>. For
reproducible results, use the same LeRobot version as the machine where
the script was tested.

### Option B --- You already have LeRobot installed in your project environment

Activate that environment, then run:

``` bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install mujoco pygame numpy
python -c "import mujoco, pygame, numpy; from lerobot.datasets import LeRobotDataset; print('Imports OK')"
```

If the import check fails with `No module named 'lerobot'`, install
LeRobot using Option A into the environment you intend to use.

## 7. Verify the Python imports

With the intended environment activated:

``` bash
python -c "import sys; print(sys.executable); import mujoco, pygame, numpy; from lerobot.datasets import LeRobotDataset; print('All imports OK')"
```

Check that `sys.executable` points to the Python inside the intended
`.venv`. Do not install packages in one environment and run the script
with another.

## 8. Verify the MuJoCo XML names

The script expects these names in `vla_scene.xml`:

-   Cameras: `workspace_camera`, `wrist_camera`
-   Cube body: `cube`, with a **free joint**
-   Site: `gripper_pinch`
-   Gripper actuator: `gripper_fingers_actuator`
-   Gripper joint: `gripper_right_driver_joint`
-   Arm joints: `shoulder_pan_joint`, `shoulder_lift_joint`,
    `elbow_joint`, `wrist_1_joint`, `wrist_2_joint`, `wrist_3_joint`
-   Arm actuators: `shoulder_pan`, `shoulder_lift`, `elbow`, `wrist_1`,
    `wrist_2`, `wrist_3`

The XML must also contain the robot, gripper, floor, lights, and all
assets referenced by the scene. If names differ, update the script or
XML consistently.

## 9. Connect and test the controller

Connect the controller before launching the script. In the active
virtual environment, run:

``` bash
python -c "import pygame; pygame.init(); pygame.joystick.init(); print('Controllers:', pygame.joystick.get_count()); [print(i, pygame.joystick.Joystick(i).get_name()) for i in range(pygame.joystick.get_count())]"
```

At least one controller should be listed. Current mapping:

  Controller input         Action
  ------------------------ ------------------
  Left stick X             Move target in X
  Left stick Y             Move target in Y
  Right stick Y (axis 3)   Move target in Z
  Button 0                 Close gripper
  Button 1                 Open gripper

Controller axis/button indices vary by model. If the inputs do not
match, adjust the mappings in `ares_record.py`.

## 10. Set the episode count and fix the uploaded script

Near the top of `ares_record.py`, set the desired values, for example:

``` python
TARGET_TOTAL_EPISODES = 30
EPISODES_PER_RUN = 5
EPISODE_TIME_S = 240.0
FPS = 30
IMAGE_SIZE = 256
```

-   `TARGET_TOTAL_EPISODES`: final desired number of episodes.
-   `EPISODES_PER_RUN`: maximum new episodes recorded each time you
    launch the script.
-   If 10 episodes already exist and the target is 30 with a batch size
    of 5, the next run should add up to five episodes.

**Required correction:** the uploaded script uses
`episodes_to_record = NUM_EPISODES` in the new-dataset `else` block, but
`NUM_EPISODES` is not defined. Replace that line with:

``` python
episodes_to_record = min(
    EPISODES_PER_RUN,
    TARGET_TOTAL_EPISODES - existing_episodes
)
```

Make this change before running the script.

## 11. Run data collection

Activate the environment in which LeRobot and the other dependencies
were installed, then move to the folder containing the script:

``` bash
source ~/lerobot/.venv/bin/activate
cd ~/Downloads/mjprojects
python ares_record.py
```

Adjust both paths if your folders differ.

At startup, check that the model loads, the controller is detected, and
the printed cube coordinates change between episodes. The MuJoCo viewer
needs a graphical desktop session and working OpenGL; a headless SSH
session may not open the viewer.

For each episode:

1.  The script resets the arm and randomizes the cube.
2.  Use the controller to move the target and align the gripper.
3.  Button 0 closes the gripper; button 1 opens it.
4.  Attempt the pick-up before the episode time limit.
5.  Check the terminal and inspect saved data/videos where available.

A saved episode is not necessarily a successful pick-up. Review the
recorded demonstrations.

## 12. Resume the dataset safely

Do not delete the dataset folder if you want to keep prior episodes.
When `DATASET_ROOT` already exists, the script is intended to call
`LeRobotDataset.resume(...)`.

Do not run two recording scripts against the same dataset folder
simultaneously. Back up the entire dataset folder before moving it to
another PC.

If the script reports the target episode count has already been reached,
increase `TARGET_TOTAL_EPISODES` intentionally if you want to record
more.

## 13. What gets saved

The script stores:

-   `observation.state`: six arm joint positions plus gripper state
-   `action`: six arm actuator targets plus gripper command
-   `observation.images.camera1`: workspace camera image
-   `observation.images.camera2`: wrist camera image
-   timestamps, frame/episode/task metadata

With `use_videos=False`, images are stored in Parquet data. The exact
folder/file layout can vary by LeRobot version.

## 14. Troubleshooting

### `python3.12: command not found`

Python 3.12 is not installed or is not on your PATH. Install it for your
Ubuntu release, then verify with `python3.12 --version`.

### `No module named venv` or `ensurepip is not available`

Install the matching venv package, for example:

``` bash
sudo apt install python3.12-venv
```

### `ModuleNotFoundError: mujoco`, `pygame`, or `lerobot`

Activate the correct `.venv` and install the missing package there.
Verify with the import command in Section 7.

### XML or mesh file not found

Check `MODEL_PATH` and preserve all relative asset paths. Linux
filenames are case-sensitive.

### Controller not detected

Check that Linux detects the controller and that it is connected before
launching the script. Try another USB port or Bluetooth pairing if
needed.

### MuJoCo viewer does not open

Run from a normal graphical desktop session, update the graphics driver,
and check OpenGL support. Remote/headless sessions may require a
separate display configuration.

### Dataset is not recording new episodes

Check the printed number of existing episodes and new episodes for this
run. If the new-episode count is zero, the configured target may already
have been reached.

## Official references

-   Python: <https://www.python.org/downloads/>
-   Ubuntu: <https://ubuntu.com/download/desktop>
-   Git: <https://git-scm.com/>
-   LeRobot: <https://github.com/huggingface/lerobot>
-   MuJoCo: <https://mujoco.readthedocs.io/>

## Quick checklist

-   [ ] Install Python 3.12.x, Git, and system dependencies.
-   [ ] Copy `ares_record.py`, `vla_scene.xml`, and all referenced
    assets.
-   [ ] Set the correct Linux `MODEL_PATH` and `DATASET_ROOT`.
-   [ ] Create and activate a Python 3.12 virtual environment.
-   [ ] Install LeRobot, `mujoco`, `pygame`, and `numpy` in that
    environment.
-   [ ] Confirm the XML names match Section 8.
-   [ ] Connect and test the controller.
-   [ ] Fix `episodes_to_record = NUM_EPISODES`.
-   [ ] Set target total and batch size.
-   [ ] Run `python ares_record.py` from a graphical desktop session.
-   [ ] Verify episodes and review the saved data.
