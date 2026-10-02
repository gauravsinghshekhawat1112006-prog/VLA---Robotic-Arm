import mujoco
import mujoco.viewer
import pygame
import numpy as np
import time
import random


# =========================================================
# MODEL
# =========================================================

MODEL_PATH = r"C:\mujoco_menagerie\universal_robots_ur5e\vla_scene.xml"


# =========================================================
# SETTINGS
# =========================================================

TARGET_SPEED = 0.08       # meters/second
IK_GAIN = 6.0             # IK correction strength
DAMPING = 0.05
MAX_JOINT_SPEED = 0.60    # rad/second
DEADZONE = 0.06

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 255.0


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading model...")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

print("Model loaded successfully!")


# =========================================================
# HELPERS
# =========================================================

def find_joint(name):
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name
    )

    if jid < 0:
        raise RuntimeError(
            f"Joint not found: {name}"
        )

    return jid


def find_actuator(name):
    aid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name
    )

    if aid < 0:
        raise RuntimeError(
            f"Actuator not found: {name}"
        )

    return aid


def clamp(value, low, high):
    return max(low, min(high, value))


# =========================================================
# ARM JOINTS
# =========================================================

joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
]

actuator_names = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3"
]

joint_ids = [
    find_joint(name)
    for name in joint_names
]

arm_actuators = [
    find_actuator(name)
    for name in actuator_names
]


print("Joint IDs:", joint_ids)
print("Actuator IDs:", arm_actuators)


# =========================================================
# GRIPPER
# =========================================================

gripper_actuator = find_actuator(
    "gripper_fingers_actuator"
)

print("Gripper actuator:", gripper_actuator)


# =========================================================
# GRIPPER PINCH SITE
# =========================================================

pinch_site = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "gripper_pinch"
)

if pinch_site < 0:
    raise RuntimeError(
        "gripper_pinch site not found!"
    )

print("Gripper pinch site:", pinch_site)


# =========================================================
# MAKE PINCH SITE VISIBLE
# =========================================================

# Make it a visible sphere
model.site_type[pinch_site] = (
    mujoco.mjtGeom.mjGEOM_SPHERE
)

# Radius
model.site_size[pinch_site] = np.array([
    0.025,
    0.025,
    0.025
])

# Bright red
model.site_rgba[pinch_site] = np.array([
    1.0,
    0.0,
    0.0,
    1.0
])

# Put it in the always-visible site group
model.site_group[pinch_site] = 0


# =========================================================
# QPOS / DOF ADDRESSES
# =========================================================

qpos_addr = np.array([
    model.jnt_qposadr[j]
    for j in joint_ids
])

dof_addr = np.array([
    model.jnt_dofadr[j]
    for j in joint_ids
])


# =========================================================
# CUBE
# =========================================================

cube_body = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "cube"
)

if cube_body < 0:
    raise RuntimeError(
        "Cube body not found!"
    )


# =========================================================
# FIND CUBE FREE JOINT
# =========================================================

cube_joint = -1

for j in range(model.njnt):

    if (
        model.jnt_bodyid[j] == cube_body
        and
        model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    ):
        cube_joint = j
        break


if cube_joint < 0:
    raise RuntimeError(
        "Cube free joint not found!"
    )


cube_qpos = model.jnt_qposadr[cube_joint]
cube_dof = model.jnt_dofadr[cube_joint]


# =========================================================
# RANDOMIZE CUBE
# =========================================================

def randomize_cube():

    x = random.uniform(0.25, 0.55)
    y = random.uniform(-0.30, 0.30)

    data.qpos[cube_qpos + 0] = x
    data.qpos[cube_qpos + 1] = y
    data.qpos[cube_qpos + 2] = 0.06

    # Quaternion
    data.qpos[cube_qpos + 3] = 1.0
    data.qpos[cube_qpos + 4] = 0.0
    data.qpos[cube_qpos + 5] = 0.0
    data.qpos[cube_qpos + 6] = 0.0

    data.qvel[
        cube_dof:cube_dof + 6
    ] = 0.0

    mujoco.mj_forward(model, data)

    print(
        f"Cube position: "
        f"x={x:.3f}, "
        f"y={y:.3f}, "
        f"z=0.060"
    )


randomize_cube()


# =========================================================
# INITIAL JOINT TARGETS
# =========================================================

joint_targets = data.qpos[
    qpos_addr
].copy()

home_q = joint_targets.copy()


# =========================================================
# VERY IMPORTANT:
# INITIALIZE THE ACTUATORS TO CURRENT POSITION
# =========================================================

for i in range(6):

    data.ctrl[
        arm_actuators[i]
    ] = joint_targets[i]


# Open gripper initially
data.ctrl[
    gripper_actuator
] = GRIPPER_OPEN


mujoco.mj_forward(
    model,
    data
)


# =========================================================
# RED DOT TARGET
# =========================================================

target_position = data.site_xpos[
    pinch_site
].copy()

target_rotation = data.site_xmat[
    pinch_site
].reshape(3, 3).copy()

print(
    "Initial red-dot position:",
    target_position
)


# =========================================================
# CONTROLLER
# =========================================================

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:

    pygame.quit()

    raise RuntimeError(
        "Controller not detected!"
    )


joystick = pygame.joystick.Joystick(0)
joystick.init()

print()
print(
    "Controller:",
    joystick.get_name()
)

print(
    "Axes:",
    joystick.get_numaxes()
)

print(
    "Buttons:",
    joystick.get_numbuttons()
)

print(
    "Hats:",
    joystick.get_numhats()
)


# =========================================================
# IK
# =========================================================

def calculate_ik(dt):

    global joint_targets

    # =====================================================
    # CURRENT END-EFFECTOR STATE
    # =====================================================

    current_position = data.site_xpos[
        pinch_site
    ].copy()

    current_rotation = data.site_xmat[
        pinch_site
    ].reshape(3, 3)


    # =====================================================
    # POSITION ERROR
    # =====================================================

    position_error = (
        target_position
        - current_position
    )


    # =====================================================
    # ORIENTATION ERROR
    #
    # Keep gripper orientation fixed to its startup
    # orientation instead of letting IK freely rotate it.
    # =====================================================

    orientation_error = 0.5 * (
        np.cross(
            current_rotation[:, 0],
            target_rotation[:, 0]
        )
        +
        np.cross(
            current_rotation[:, 1],
            target_rotation[:, 1]
        )
        +
        np.cross(
            current_rotation[:, 2],
            target_rotation[:, 2]
        )
    )


    # =====================================================
    # TASK GAINS
    # =====================================================

    POSITION_GAIN = 5.0
    ORIENTATION_GAIN = 2.0


    desired_linear_velocity = (
        POSITION_GAIN
        * position_error
    )

    desired_angular_velocity = (
        ORIENTATION_GAIN
        * orientation_error
    )


    # =====================================================
    # LIMIT CARTESIAN SPEED
    # =====================================================

    max_linear_speed = 0.20

    linear_speed = np.linalg.norm(
        desired_linear_velocity
    )

    if linear_speed > max_linear_speed:

        desired_linear_velocity *= (
            max_linear_speed
            /
            linear_speed
        )


    # =====================================================
    # 6D TASK VELOCITY
    # =====================================================

    desired_velocity = np.concatenate(
        (
            desired_linear_velocity,
            desired_angular_velocity
        )
    )


    # =====================================================
    # JACOBIAN
    # =====================================================

    jacp = np.zeros(
        (3, model.nv)
    )

    jacr = np.zeros(
        (3, model.nv)
    )

    mujoco.mj_jacSite(
        model,
        data,
        jacp,
        jacr,
        pinch_site
    )


    J = np.vstack(
        (
            jacp[:, dof_addr],
            jacr[:, dof_addr]
        )
    )


    # =====================================================
    # DAMPED PSEUDO-INVERSE
    # =====================================================

    lambda_value = 0.08

    J_pinv = (
        J.T
        @
        np.linalg.inv(
            J @ J.T
            +
            lambda_value
            * lambda_value
            * np.eye(6)
        )
    )


    # =====================================================
    # MAIN IK MOTION
    # =====================================================

    dq_task = (
        J_pinv
        @
        desired_velocity
    )


    # =====================================================
    # NULL-SPACE HOME POSTURE
    #
    # This stops the arm from folding into strange
    # configurations while still allowing X/Y/Z motion.
    # =====================================================

    current_q = data.qpos[
        qpos_addr
    ].copy()


    HOME_GAIN = 0.12


    dq_home = (
        HOME_GAIN
        *
        (home_q - current_q)
    )


    null_space = (
        np.eye(6)
        -
        J_pinv @ J
    )


    dq_null = (
        null_space
        @
        dq_home
    )


    # =====================================================
    # FINAL JOINT VELOCITY
    # =====================================================

    dq = (
        dq_task
        +
        dq_null
    )


    # =====================================================
    # JOINT SPEED LIMIT
    # =====================================================

    MAX_JOINT_SPEED = 0.45

    dq = np.clip(
        dq,
        -MAX_JOINT_SPEED,
        MAX_JOINT_SPEED
    )


    # =====================================================
    # UPDATE JOINT TARGETS
    # =====================================================

    new_targets = (
        joint_targets
        +
        dq * dt
    )


    # =====================================================
    # JOINT LIMITS
    # =====================================================

    for i, jid in enumerate(joint_ids):

        lower = model.jnt_range[
            jid,
            0
        ]

        upper = model.jnt_range[
            jid,
            1
        ]

        new_targets[i] = clamp(
            new_targets[i],
            lower,
            upper
        )


    joint_targets = new_targets


# =========================================================
# CAMERA FOLLOW
# =========================================================

def update_camera(viewer):

    gripper_pos = data.site_xpos[
        pinch_site
    ].copy()

    cube_pos = data.xpos[
        cube_body
    ].copy()

    focus = (
        0.70 * gripper_pos
        +
        0.30 * cube_pos
    )

    viewer.cam.lookat[:] = (
        0.94 * viewer.cam.lookat
        +
        0.06 * focus
    )


# =========================================================
# VIEWER
# =========================================================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    # Make all site groups visible
    viewer.opt.sitegroup[:] = 1

    viewer.cam.type = (
        mujoco.mjtCamera.mjCAMERA_FREE
    )

    viewer.cam.distance = 0.90
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -25

    # Make the initial camera target the robot
    viewer.cam.lookat[:] = data.site_xpos[
        pinch_site
    ]

    print()
    print("======================================")
    print("IK TELEOPERATION")
    print("======================================")
    print()
    print("LEFT STICK X  -> X")
    print("LEFT STICK Y  -> Y")
    print("RIGHT STICK Y -> Z")
    print()
    print("BUTTON 0 -> CLOSE GRIPPER")
    print("BUTTON 1 -> OPEN GRIPPER")
    print()
    print("RED SPHERE = GRIPPER CONTROL POINT")
    print("======================================")
    print()

    last_time = time.perf_counter()

    while viewer.is_running():

        # =================================================
        # TIME
        # =================================================

        now = time.perf_counter()

        dt = now - last_time

        last_time = now

        if dt > 0.03:
            dt = 0.03


        # =================================================
        # CONTROLLER
        # =================================================

        pygame.event.pump()

        left_x = joystick.get_axis(0)
        left_y = joystick.get_axis(1)
        right_y = joystick.get_axis(3)


        # =================================================
        # DEADZONE
        # =================================================

        if abs(left_x) < DEADZONE:
            left_x = 0.0

        if abs(left_y) < DEADZONE:
            left_y = 0.0

        if abs(right_y) < DEADZONE:
            right_y = 0.0


        # =================================================
        # MOVE RED-DOT TARGET
        # =================================================

        target_position[0] += (
            left_x
            * TARGET_SPEED
            * dt
        )

        target_position[1] += (
            -left_y
            * TARGET_SPEED
            * dt
        )

        target_position[2] += (
            -right_y
            * TARGET_SPEED
            * dt
        )


        # =================================================
        # WORKSPACE LIMITS
        # =================================================

        target_position[0] = clamp(
            target_position[0],
            0.15,
            0.70
        )

        target_position[1] = clamp(
            target_position[1],
            -0.55,
            0.55
        )

        target_position[2] = clamp(
            target_position[2],
            0.12,
            0.85
        )


        # =================================================
        # IK
        # =================================================

        calculate_ik(dt)


        # =================================================
        # SEND JOINT TARGETS
        # =================================================

        for i in range(6):

            data.ctrl[
                arm_actuators[i]
            ] = joint_targets[i]


        # =================================================
        # GRIPPER
        # =================================================

        if joystick.get_button(0):

            data.ctrl[
                gripper_actuator
            ] = GRIPPER_CLOSED


        if joystick.get_button(1):

            data.ctrl[
                gripper_actuator
            ] = GRIPPER_OPEN


        # =================================================
        # STEP
        # =================================================

        mujoco.mj_step(
            model,
            data
        )


        # =================================================
        # CAMERA
        # =================================================

        update_camera(viewer)


        # =================================================
        # VIEWER
        # =================================================

        viewer.sync()

        time.sleep(0.001)


pygame.quit()