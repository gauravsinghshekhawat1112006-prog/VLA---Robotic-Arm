import time
import numpy as np
import mujoco
import mujoco.viewer
import pygame

MODEL_PATH = r"C:\mujoco_menagerie\universal_robots_ur5e\vla_scene.xml"
TARGET_SPEED = 0.25
TARGET_Z_SPEED = 0.20
TARGET_RADIUS = 0.025
IK_ITERS = 8
IK_DAMPING = 0.08
IK_STEP_LIMIT = 0.12
DEADZONE = 0.10
GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 255.0
TARGET_X_MIN, TARGET_X_MAX = -0.75, 0.75
TARGET_Y_MIN, TARGET_Y_MAX = -0.75, 0.75
TARGET_Z_MIN, TARGET_Z_MAX = 0.05, 1.15

print("Loading model...")
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
print("Model loaded successfully!")

def find_actuator(name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if i < 0: raise RuntimeError(f"Actuator not found: {name}")
    return i

def find_joint(name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if i < 0: raise RuntimeError(f"Joint not found: {name}")
    return i

def find_site(name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if i < 0: raise RuntimeError(f"Site not found: {name}")
    return i

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

arm_actuators = [find_actuator(n) for n in (
    "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")]
gripper_actuator = find_actuator("gripper_fingers_actuator")
arm_joints = [find_joint(n) for n in (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")]
arm_qpos = np.array([model.jnt_qposadr[j] for j in arm_joints], dtype=int)
ee_site = find_site("gripper_pinch")

mujoco.mj_forward(model, data)
q_target = data.qpos[arm_qpos].copy()
target_pos = data.site_xpos[ee_site].copy()
gripper_target = GRIPPER_OPEN

def solve_ik(target, q_start):
    q = q_start.copy()
    for _ in range(IK_ITERS):
        data.qpos[arm_qpos] = q
        mujoco.mj_forward(model, data)
        err = target - data.site_xpos[ee_site]
        if np.linalg.norm(err) < 0.001:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, ee_site)
        J = jacp[:, arm_qpos]
        dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING**2 * np.eye(3), err)
        n = np.linalg.norm(dq)
        if n > IK_STEP_LIMIT:
            dq *= IK_STEP_LIMIT / n
        q += dq
        for k, joint in enumerate(arm_joints):
            q[k] = clamp(q[k], model.jnt_range[joint, 0], model.jnt_range[joint, 1])
    return q

pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    print("Controller not detected!")
    pygame.quit()
    raise SystemExit
joystick = pygame.joystick.Joystick(0)
joystick.init()
print("Controller:", joystick.get_name())
print("\n3D TARGET / IK CONTROLLER")
print("Left Stick X  -> target X")
print("Left Stick Y  -> target Y")
print("D-Pad Up/Down -> target Z")
print("Button 0      -> gripper CLOSE")
print("Button 1      -> gripper OPEN")
print("The sphere is visual only; it has NO physics.")


def update_target_visual(viewer):
    viewer.user_scn.ngeom = 0
    geom = viewer.user_scn.geoms[0]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([TARGET_RADIUS, 0.0, 0.0]),
        target_pos,
        np.eye(3).flatten(),
        np.array([0.1, 1.0, 0.1, 1.0]),
    )
    viewer.user_scn.ngeom = 1

with mujoco.viewer.launch_passive(model, data) as viewer:
    update_target_visual(viewer)
    last_time = time.perf_counter()
    while viewer.is_running():
        now = time.perf_counter()
        dt = min(now - last_time, 0.05)
        last_time = now
        pygame.event.pump()

        lx = joystick.get_axis(0)
        ly = joystick.get_axis(1)
        if abs(lx) < DEADZONE: lx = 0.0
        if abs(ly) < DEADZONE: ly = 0.0

        target_pos[0] += lx * TARGET_SPEED * dt
        target_pos[1] += -ly * TARGET_SPEED * dt

        if joystick.get_numhats() > 0:
            _, hy = joystick.get_hat(0)
            target_pos[2] += hy * TARGET_Z_SPEED * dt

        target_pos[0] = clamp(target_pos[0], TARGET_X_MIN, TARGET_X_MAX)
        target_pos[1] = clamp(target_pos[1], TARGET_Y_MIN, TARGET_Y_MAX)
        target_pos[2] = clamp(target_pos[2], TARGET_Z_MIN, TARGET_Z_MAX)

        q_target = solve_ik(target_pos, q_target)
        for i, actuator in enumerate(arm_actuators):
            data.ctrl[actuator] = q_target[i]

        if joystick.get_button(0):
            gripper_target = GRIPPER_CLOSE
        if joystick.get_button(1):
            gripper_target = GRIPPER_OPEN
        data.ctrl[gripper_actuator] = gripper_target

        mujoco.mj_step(model, data)
        update_target_visual(viewer)
        viewer.sync()
        time.sleep(0.005)

pygame.quit()
print("Simulation closed.")
