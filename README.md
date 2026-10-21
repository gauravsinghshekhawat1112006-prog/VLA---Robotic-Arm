# ARES UR5e + Robotiq MuJoCo Data Collection

This README explains how to set up and run `ares_record.py` on a new Windows PC. The instructions are based on the uploaded script. They assume you have the MuJoCo UR5e scene and the Robotiq gripper assets already integrated into `vla_scene.xml`.

> **Important:** `ares_record.py` does not contain the robot model itself. You must copy the complete MuJoCo scene and every referenced XML/mesh asset to the new computer. The XML must include the names expected by the script (listed below).

## 1. Requirements

- Windows 10/11, 64-bit
- Python **3.12.x** (the working setup used Python 3.12.9)
- Git
- An NVIDIA/AMD/Intel graphics driver that supports OpenGL
- MuJoCo scene XML and all referenced assets
- An Xbox-compatible controller (or another controller whose axis/button mapping you configure)
- Internet connection for installing Python packages

The script uses:
- `mujoco` and `mujoco.viewer` — simulation and interactive viewer
- `pygame` — controller input
- `numpy` — numerical operations
- `lerobot.datasets.LeRobotDataset` — dataset creation/resume and episode writing
- Python standard-library modules: `time`, `pathlib`, and `random`

## 2. Install Python 3.12

1. Open the official Python downloads page: <https://www.python.org/downloads/>.
2. Download a **Python 3.12.x Windows installer (64-bit)**.
3. Run the installer.
4. Tick **Add python.exe to PATH**, then choose **Install Now**.
5. Open a *new* PowerShell window and verify:

```powershell
py -3.12 --version
```

It should print `Python 3.12.x`. If `py` is not found, rerun the installer and enable the Python Launcher/PATH options.

## 3. Install Git

Download and install Git for Windows: <https://git-scm.com/download/win>.

Verify in a new PowerShell window:

```powershell
git --version
```

## 4. Put the project files in one place

Example folder layout:

```text
C:\ARES\
├── ares_record.py
├── vla_scene.xml
└── ...all XML/mesh assets referenced by vla_scene.xml...
```

Keep the original directory structure of the referenced assets. Do not copy only `vla_scene.xml`; missing meshes/includes will prevent MuJoCo from loading the model.

### Check the model path

Open `ares_record.py` and update `MODEL_PATH` to the actual location of the scene XML. For the example layout above:

```python
MODEL_PATH = r"C:\ARES\vla_scene.xml"
```

The uploaded file currently contains a Linux path (`/home/mars/...`), so it must be changed on a Windows PC.

### Check the dataset path

Set `DATASET_ROOT` to the desired dataset folder. For example:

```python
DATASET_ROOT = Path(r"C:\ARES\datasets\ares_ur5e_pick_cube")
```

Use a different path if preferred. Make sure the parent directory exists and you have write permission.

## 5. Create and activate a virtual environment

In PowerShell:

```powershell
cd C:\ARES
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this for the current PowerShell session and try activating again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Confirm the environment is active: the prompt should begin with `(.venv)`.

Upgrade the packaging tools:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

## 6. Install LeRobot and the required libraries

The script imports LeRobot directly. The most reproducible approach is to use the same LeRobot source version as the machine where the script was tested.

### Option A — if you have the LeRobot repository

Clone the official repository if you do not already have it:

```powershell
cd C:\ARES
git clone https://github.com/huggingface/lerobot.git
cd .\lerobot
```

Create/activate the environment **in the LeRobot repository** (or use the existing `.venv` there), then install the repository in editable mode:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

Move/copy `ares_record.py` and the MuJoCo scene into your project folder, but run the script with this LeRobot environment activated. If the repository's current installation instructions differ, follow the official LeRobot installation guide: <https://github.com/huggingface/lerobot>.

Install/verify the script's direct dependencies in that same environment:

```powershell
python -m pip install mujoco pygame numpy
```

### Option B — if LeRobot is already installed in your project environment

Activate that environment and run:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install mujoco pygame numpy
python -c "import mujoco, pygame, numpy; from lerobot.datasets import LeRobotDataset; print('Imports OK')"
```

If the last command fails at `lerobot`, install LeRobot using Option A. Do not install packages into a different Python environment by mistake.

> **PyTorch/TorchCodec note:** the data-collection script does not explicitly import TorchCodec. LeRobot may print a TorchCodec/FFmpeg warning and fall back to another decoder. If the script continues and records episodes, this warning alone does not necessarily stop collection. Fix it only if an operation you need actually fails.

## 7. Verify the MuJoCo scene before recording

Open `vla_scene.xml` and confirm the scene contains the names expected by the script:

- Camera: `workspace_camera`
- Camera: `wrist_camera`
- Cube body: `cube` (with a **free joint**)
- Site: `gripper_pinch`
- Gripper actuator: `gripper_fingers_actuator`
- Gripper joint: `gripper_right_driver_joint`
- Arm joints:
  - `shoulder_pan_joint`
  - `shoulder_lift_joint`
  - `elbow_joint`
  - `wrist_1_joint`
  - `wrist_2_joint`
  - `wrist_3_joint`
- Arm actuators:
  - `shoulder_pan`
  - `shoulder_lift`
  - `elbow`
  - `wrist_1`
  - `wrist_2`
  - `wrist_3`

The XML also needs to define the robot, gripper, floor, lights, and any meshes/materials referenced by the scene. The script will stop with a clear error if several expected names are missing.

## 8. Check controller detection

Connect the controller before running the script. With the virtual environment active, test:

```powershell
python -c "import pygame; pygame.init(); pygame.joystick.init(); print('Controllers:', pygame.joystick.get_count()); [print(i, pygame.joystick.Joystick(i).get_name()) for i in range(pygame.joystick.get_count())]"
```

At least one controller should be listed. The script's current mapping is:

| Controller input | Action |
|---|---|
| Left stick X | Move target in X |
| Left stick Y | Move target in Y |
| Right stick Y (axis 3) | Move target in Z |
| Button 0 | Close gripper |
| Button 1 | Open gripper |

Different controller models can use different axis/button indices. If your controller behaves differently, the mappings in `ares_record.py` must be adjusted.

## 9. Configure episode count and batch size

Near the top of `ares_record.py`, set:

```python
TARGET_TOTAL_EPISODES = 30
EPISODES_PER_RUN = 5
EPISODE_TIME_S = 240.0
FPS = 30
IMAGE_SIZE = 256
```

- `TARGET_TOTAL_EPISODES` is the desired final number of episodes in the dataset.
- `EPISODES_PER_RUN` is the maximum number of new episodes to record each time you launch the script.
- Example: with 10 episodes already saved, the next run records episodes 11–15. The last run records fewer if only a few episodes remain.

### Required code correction in the uploaded script

In the `else:` block that creates a new dataset, the uploaded script currently has:

```python
episodes_to_record = NUM_EPISODES
```

`NUM_EPISODES` is not defined in the uploaded file. Replace that line with:

```python
episodes_to_record = min(
    EPISODES_PER_RUN,
    TARGET_TOTAL_EPISODES - existing_episodes
)
```

This makes the first run follow the same batch-size rule as later runs.

## 10. Run data collection

From the folder containing `ares_record.py`, with the correct virtual environment activated:

```powershell
python ares_record.py
```

On startup, verify:
- `Model loaded successfully!`
- The correct controller is detected
- The printed cube coordinates are within the configured X/Y ranges
- The dataset's existing episode count and number of new episodes look correct

The MuJoCo viewer should open. For each episode:

1. The script resets the arm and randomizes the cube.
2. Check the printed cube position; it should usually differ between episodes.
3. Use the controller to move the red target dot and align the gripper.
4. Press button 0 to close the gripper and button 1 to open it.
5. Start moving the controller to begin recording.
6. Try to complete the pick-up. The script saves the episode when its cube-lift detection succeeds, or when the episode time limit is reached.

**Important:** a saved episode is not automatically a successful demonstration. Check the recorded videos/frames and keep track of whether the robot actually picked up the cube.

The script records two images per frame:
- `observation.images.camera1`: workspace camera
- `observation.images.camera2`: wrist camera

It also records:
- `observation.state`: six arm joint positions plus gripper state
- `action`: six arm actuator targets plus gripper command
- timestamp, frame index, episode index, task index, and task text

## 11. Resume an existing dataset

Do not delete the dataset folder if you want to keep previously recorded episodes. The script uses `LeRobotDataset.resume(...)` when `DATASET_ROOT` already exists.

For example, if the dataset contains 5 episodes and the target is 30 with a batch size of 5, the next run should add episodes 6–10. Each later run adds up to five until the target is reached.

If the script prints `Existing episodes: 30` and `This run: 0 new episodes`, it has reached the target; no additional episodes will be recorded.

**Do not run two recording scripts against the same dataset folder at the same time.**

## 12. Where the data is saved

The dataset is saved under `DATASET_ROOT`, for example:

```text
C:\ARES\datasets\ares_ur5e_pick_cube\
├── data\
├── meta\
└── videos\   (may be absent/empty when use_videos=False)
```

With `use_videos=False`, images are stored in the dataset's Parquet data rather than relying on a video file for each camera. The exact internal file layout can vary by LeRobot version.

Back up the whole dataset folder before making changes or moving it to another computer.

## 13. Common problems

### `ModuleNotFoundError: No module named 'mujoco'` (or `pygame` / `lerobot`)
The wrong environment is active or the package is not installed there. Activate `.venv`, then install the missing dependency using the commands above.

### `ModelPath ... not found` / XML load error
Check `MODEL_PATH`, then verify that all XML includes and mesh files are present at their expected relative paths.

### `Controller not detected!`
Connect the controller before starting the script. Check Windows sees it, then rerun the controller detection test.

### `Joint not found`, `Actuator not found`, or `Cube free joint not found`
The XML names do not match the script, or the cube is not defined with a free joint. Use the names listed in Section 7.

### Dataset already exists
This is expected when resuming. The script should call `LeRobotDataset.resume`, not `LeRobotDataset.create`, for an existing dataset. Never delete the folder unless you intentionally want to discard it.

### Viewer does not open / rendering error
Update your graphics driver and ensure the machine supports OpenGL. Try launching the script from a normal desktop session rather than a headless terminal/remote session.

### Script records no episodes
Check `This run: ... new episodes`. If it is zero, the target has already been reached or the target count is configured incorrectly.

## 14. Official references

- Python downloads: <https://www.python.org/downloads/>
- Git for Windows: <https://git-scm.com/download/win>
- LeRobot repository and installation instructions: <https://github.com/huggingface/lerobot>
- MuJoCo documentation: <https://mujoco.readthedocs.io/>

---

## Quick-start checklist

- [ ] Install Python 3.12.x and Git.
- [ ] Copy `ares_record.py`, `vla_scene.xml`, and all referenced assets.
- [ ] Update `MODEL_PATH` and `DATASET_ROOT`.
- [ ] Create and activate `.venv`.
- [ ] Install LeRobot in that environment.
- [ ] Install `mujoco`, `pygame`, and `numpy`.
- [ ] Confirm the XML names match Section 7.
- [ ] Connect and test the controller.
- [ ] Fix `episodes_to_record = NUM_EPISODES` as described in Section 9.
- [ ] Set target total and episodes per run.
- [ ] Run `python ares_record.py`.
- [ ] Verify saved episodes and review the recorded data.
