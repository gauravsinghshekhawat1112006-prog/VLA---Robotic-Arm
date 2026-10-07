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

# Speed of the RED TARGET DOT
TARGET_SPEED = 0.12       # meters / second

# IK response
IK_GAIN = 5.0

# Damped IK
DAMPING = 0.06

# Maximum joint speed
MAX_JOINT_SPEED = 0.65   # rad / second

# Joystick deadzone
DEADZONE = 0.06

# Gripper
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 255.0


# =========================================================
# STARTING POSE
# =========================================================
#
# Upright configuration.
# Gripper should point approximately downward.
#

START_Q = np.array(
    [
        0.0,       # shoulder pan
        -1.20,     # shoulder lift
        1.65,      # elbow
        -2.02,     # wrist 1
        -1.57,     # wrist 2
        0.0        # wrist 3
    ],
    dtype=np.float64
)


# =========================================================
# CUBE RANDOM RANGE
# =========================================================

CUBE_X_MIN = 0.25
CUBE_X_MAX = 0.55

CUBE_Y_MIN = -0.30
CUBE_Y_MAX = 0.30

CUBE_Z = 0.06


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading model...")

model = mujoco.MjModel.from_xml_path(
    MODEL_PATH
)

data = mujoco.MjData(model)

print("Model loaded successfully!")


# =========================================================
# HELPERS
# =========================================================

def find_joint(name):

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name
    )

    if joint_id < 0:
        raise RuntimeError(
            f"Joint not found: {name}"
        )

    return joint_id


def find_actuator(name):

    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name
    )

    if actuator_id < 0:
        raise RuntimeError(
            f"Actuator not found: {name}"
        )

    return actuator_id


def clamp(value, low, high):

    return max(
        low,
        min(high, value)
    )


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


# =========================================================
# GRIPPER
# =========================================================

gripper_actuator = find_actuator(
    "gripper_fingers_actuator"
)


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


# =========================================================
# JOINT ADDRESSES
# =========================================================

qpos_addr = np.array(

    [
        model.jnt_qposadr[j]
        for j in joint_ids
    ],

    dtype=np.int32

)


dof_addr = np.array(

    [
        model.jnt_dofadr[j]
        for j in joint_ids
    ],

    dtype=np.int32

)


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

        model.jnt_type[j]
        == mujoco.mjtJoint.mjJNT_FREE

    ):

        cube_joint = j
        break


if cube_joint < 0:

    raise RuntimeError(
        "Cube free joint not found!"
    )


cube_qpos = model.jnt_qposadr[
    cube_joint
]

cube_dof = model.jnt_dofadr[
    cube_joint
]


# =========================================================
# RANDOMIZE CUBE
# =========================================================

def randomize_cube():

    x = random.uniform(
        CUBE_X_MIN,
        CUBE_X_MAX
    )

    y = random.uniform(
        CUBE_Y_MIN,
        CUBE_Y_MAX
    )


    data.qpos[
        cube_qpos + 0
    ] = x

    data.qpos[
        cube_qpos + 1
    ] = y

    data.qpos[
        cube_qpos + 2
    ] = CUBE_Z


    # quaternion
    data.qpos[
        cube_qpos + 3
    ] = 1.0

    data.qpos[
        cube_qpos + 4
    ] = 0.0

    data.qpos[
        cube_qpos + 5
    ] = 0.0

    data.qpos[
        cube_qpos + 6
    ] = 0.0


    # zero cube velocity

    data.qvel[
        cube_dof:cube_dof + 6
    ] = 0.0


    print(
        f"Cube position: "
        f"x={x:.3f}, "
        f"y={y:.3f}, "
        f"z={CUBE_Z:.3f}"
    )


# =========================================================
# SET INITIAL ARM POSE
# =========================================================

# Put the arm directly into the desired pose.
# There is NO movement toward this pose.

data.qpos[
    qpos_addr
] = START_Q


# Zero arm velocities

data.qvel[
    dof_addr
] = 0.0


# Randomize cube BEFORE forward kinematics

randomize_cube()


# =========================================================
# SET ACTUATOR TARGETS
# =========================================================

for i in range(6):

    data.ctrl[
        arm_actuators[i]
    ] = START_Q[i]


# Gripper open

data.ctrl[
    gripper_actuator
] = GRIPPER_OPEN


# =========================================================
# FORWARD KINEMATICS
# =========================================================

mujoco.mj_forward(
    model,
    data
)


# =========================================================
# HOME POSTURE
# =========================================================

joint_targets = START_Q.copy()


# =========================================================
# INITIAL RED TARGET
# =========================================================

target_position = data.site_xpos[
    pinch_site
].copy()


# Keep the starting orientation fixed

target_rotation = data.site_xmat[
    pinch_site
].reshape(3, 3).copy()


print()
print(
    "Initial gripper position:",
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
# IK FUNCTION
# =========================================================

def calculate_ik(dt):

    global joint_targets


    # =====================================================
    # CURRENT END EFFECTOR
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
        -
        current_position

    )


    # =====================================================
    # ORIENTATION ERROR
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
    # CARTESIAN VELOCITY
    # =====================================================

    linear_velocity = (

        IK_GAIN
        *
        position_error

    )


    angular_velocity = (

        2.0
        *
        orientation_error

    )


    # Limit linear speed

    max_linear_velocity = 0.20

    linear_norm = np.linalg.norm(
        linear_velocity
    )


    if linear_norm > max_linear_velocity:

        linear_velocity *= (

            max_linear_velocity
            /
            linear_norm

        )


    # =====================================================
    # 6D TASK VELOCITY
    # =====================================================

    desired_velocity = np.concatenate(

        [
            linear_velocity,
            angular_velocity
        ]

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

        [
            jacp[:, dof_addr],
            jacr[:, dof_addr]
        ]

    )


    # =====================================================
    # DAMPED PSEUDO INVERSE
    # =====================================================

    JJt = J @ J.T


    J_pinv = (

        J.T
        @
        np.linalg.inv(

            JJt

            +

            (DAMPING ** 2)
            *
            np.eye(6)

        )

    )


    # =====================================================
    # TASK MOTION
    # =====================================================

    dq_task = (

        J_pinv
        @
        desired_velocity

    )


    # =====================================================
    # FINAL JOINT VELOCITY
    # =====================================================

    dq = dq_task


    # =====================================================
    # JOINT SPEED LIMIT
    # =====================================================

    dq = np.clip(

        dq,

        -MAX_JOINT_SPEED,
        MAX_JOINT_SPEED

    )


    # =====================================================
    # POSITION UPDATE
    # =====================================================

    joint_targets = (

        joint_targets
        +
        dq * dt

    )


    # =====================================================
    # JOINT LIMITS
    # =====================================================

    for i, jid in enumerate(
        joint_ids
    ):

        low = model.jnt_range[
            jid,
            0
        ]

        high = model.jnt_range[
            jid,
            1
        ]


        joint_targets[i] = clamp(

            joint_targets[i],
            low,
            high

        )


# =========================================================
# DRAW RED TARGET
# =========================================================

def draw_target_dot(viewer):

    viewer.user_scn.ngeom = 1

    geom = viewer.user_scn.geoms[0]


    mujoco.mjv_initGeom(

        geom,

        mujoco.mjtGeom.mjGEOM_SPHERE,

        np.array([
            0.01,
            0.01,
            0.01
        ]),

        target_position,

        np.eye(3).flatten(),

        np.array([
            1.0,
            0.0,
            0.0,
            1.0
        ])

    )


# =========================================================
# CAMERA
# =========================================================

def update_camera(viewer):

    gripper_position = data.site_xpos[
        pinch_site
    ].copy()


    cube_position = data.xpos[
        cube_body
    ].copy()


    focus = (

        0.70
        *
        gripper_position

        +

        0.30
        *
        cube_position

    )


    viewer.cam.lookat[:] = (

        0.95
        *
        viewer.cam.lookat

        +

        0.05
        *
        focus

    )


# =========================================================
# VIEWER
# =========================================================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:


    viewer.cam.type = (
        mujoco.mjtCamera.mjCAMERA_FREE
    )

    viewer.cam.distance = 0.90

    viewer.cam.azimuth = 135

    viewer.cam.elevation = -25

    viewer.cam.lookat[:] = (
        target_position
    )


    print()
    print(
        "======================================"
    )

    print(
        "IK TELEOPERATION READY"
    )

    print(
        "======================================"
    )

    print(
        "LEFT STICK X  -> X"
    )

    print(
        "LEFT STICK Y  -> Y"
    )

    print(
        "RIGHT STICK Y -> Z"
    )

    print()

    print(
        "BUTTON 0 -> CLOSE GRIPPER"
    )

    print(
        "BUTTON 1 -> OPEN GRIPPER"
    )

    print()

    print(
        "RED DOT = TARGET POSITION"
    )

    print(
        "======================================"
    )


    # =====================================================
    # IMPORTANT:
    # PHYSICS IS NOT STARTED YET
    # =====================================================

    simulation_started = False

    last_time = time.perf_counter()


    # =====================================================
    # MAIN LOOP
    # =====================================================

    while viewer.is_running():


        # =================================================
        # CONTROLLER EVENTS
        # =================================================

        now = time.perf_counter()

        pygame.event.pump()


        # =================================================
        # READ CONTROLLER
        # =================================================

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
        # DETECT REAL INPUT
        # =================================================

        controller_input = (

            abs(left_x) > DEADZONE

            or

            abs(left_y) > DEADZONE

            or

            abs(right_y) > DEADZONE

            or

            joystick.get_button(0)

            or

            joystick.get_button(1)

        )


        if controller_input:

            simulation_started = True


        # =================================================
        # DRAW RED DOT
        # =================================================

        draw_target_dot(
            viewer
        )


        # =================================================
        # BEFORE FIRST INPUT
        # =================================================

        if not simulation_started:

            # Keep exact starting pose

            data.qpos[
                qpos_addr
            ] = START_Q

            data.qvel[
                dof_addr
            ] = 0.0


            for i in range(6):

                data.ctrl[
                    arm_actuators[i]
                ] = START_Q[i]


            data.ctrl[
                gripper_actuator
            ] = GRIPPER_OPEN


            # Recalculate positions
            # WITHOUT ADVANCING PHYSICS

            mujoco.mj_forward(
                model,
                data
            )


            viewer.sync()

            time.sleep(0.01)
            last_time = time.perf_counter()

            continue


        # =================================================
        # REAL DT
        # =================================================

        dt = (
            now
            -
            last_time
        )

        last_time = now


        if dt > 0.03:

            dt = 0.03


        # =================================================
        # MOVE RED TARGET
        # =================================================

        target_position[0] += (

            left_x
            *
            TARGET_SPEED
            *
            dt

        )


        target_position[1] += (

            -left_y
            *
            TARGET_SPEED
            *
            dt

        )


        target_position[2] += (

            -right_y
            *
            TARGET_SPEED
            *
            dt

        )


        # =================================================
        # WORKSPACE LIMITS
        # =================================================

        target_position[0] = clamp(

            target_position[0],

            -0.70,
            0.70

        )


        target_position[1] = clamp(

            target_position[1],

            -0.55,
            0.55

        )


        target_position[2] = clamp(

            target_position[2],

            0.03,
            0.85

        )


        # =================================================
        # IK
        # =================================================

        calculate_ik(dt)


        # =================================================
        # SEND ARM TARGETS
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
        # STEP PHYSICS
        # =================================================

        mujoco.mj_step(
            model,
            data
        )


        # =================================================
        # CAMERA
        # =================================================

        update_camera(
            viewer
        )


        # =================================================
        # VIEWER
        # =================================================

        viewer.sync()

        time.sleep(0.001)


pygame.quit()