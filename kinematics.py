"""
Forward kinematics, geometric Jacobian and a damped-least-squares IK solver for
the 6-DOF arm defined in assets/arm6dof.xml, written with plain numpy
(homogeneous transforms).  MuJoCo computes the same thing internally; this file
exists so the math is explicit and testable (tests/test_env.py checks that
`fk()` matches MuJoCo's end-effector site to ~1e-9 m).

Convention
----------
Every link frame i is obtained from frame i-1 by
    T_i = T_{i-1} · Trans(p_i) · Rot(axis_i, q_i)
where p_i is the fixed offset of joint i expressed in frame i-1 (the <body pos>
in the XML) and axis_i is the joint axis in the local frame.  All body frames
in the XML have identity orientation, so no extra fixed rotations are needed.
The end-effector point is a final fixed offset from frame 6.
"""
from __future__ import annotations

import numpy as np

# (offset of joint frame in parent frame, joint axis in local frame)
LINKS = [
    (np.array([0.0, 0.0, 0.06]), np.array([0.0, 0.0, 1.0])),  # j1 base yaw
    (np.array([0.0, 0.0, 0.12]), np.array([0.0, 1.0, 0.0])),  # j2 shoulder pitch
    (np.array([0.0, 0.0, 0.30]), np.array([0.0, 1.0, 0.0])),  # j3 elbow pitch
    (np.array([0.0, 0.0, 0.25]), np.array([0.0, 0.0, 1.0])),  # j4 forearm roll
    (np.array([0.0, 0.0, 0.06]), np.array([0.0, 1.0, 0.0])),  # j5 wrist pitch
    (np.array([0.0, 0.0, 0.06]), np.array([0.0, 0.0, 1.0])),  # j6 wrist roll
]
EE_OFFSET = np.array([0.0, 0.0, 0.06])  # tool-centre point in link6 frame

JOINT_LOWER = np.array([-np.pi, -1.75, -2.6, -np.pi, -2.1, -np.pi])
JOINT_UPPER = np.array([np.pi, 1.75, 2.6, np.pi, 2.1, np.pi])
HOME_QPOS = np.array([0.0, 0.5, 1.2, 0.0, 0.9, 0.0])

SHOULDER = LINKS[0][0] + LINKS[1][0]  # (0, 0, 0.18) – centre of the reachable sphere
MAX_REACH = float(sum(np.linalg.norm(p) for p, _ in LINKS[2:]) + np.linalg.norm(EE_OFFSET))  # 0.73 m


def rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' formula: 3x3 rotation of `angle` about unit `axis`."""
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def homogeneous(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def fk_all(q: np.ndarray) -> list[np.ndarray]:
    """Return the list of 4x4 world transforms of link frames 1..6 plus the EE frame."""
    T = np.eye(4)
    frames = []
    for (offset, axis), qi in zip(LINKS, q):
        T = T @ homogeneous(np.eye(3), offset) @ homogeneous(rot(axis, qi), np.zeros(3))
        frames.append(T.copy())
    frames.append(T @ homogeneous(np.eye(3), EE_OFFSET))
    return frames


def fk(q: np.ndarray) -> np.ndarray:
    """End-effector position (x, y, z) in metres for joint vector q (6,)."""
    return fk_all(q)[-1][:3, 3]


def jacobian(q: np.ndarray) -> np.ndarray:
    """3x6 positional geometric Jacobian: column i = z_i x (p_ee - p_i)."""
    frames = fk_all(q)
    p_ee = frames[-1][:3, 3]
    J = np.zeros((3, 6))
    for i, (_, axis) in enumerate(LINKS):
        R, p = frames[i][:3, :3], frames[i][:3, 3]
        z = R @ axis
        J[:, i] = np.cross(z, p_ee - p)
    return J


def ik(target: np.ndarray, q0: np.ndarray | None = None, iters: int = 200,
       damping: float = 0.05, tol: float = 1e-3) -> tuple[np.ndarray, float]:
    """Damped-least-squares IK (position only, obstacle-unaware).

    Used only as a sanity tool (e.g. to show sampled targets are reachable);
    the RL agent never sees it.  Returns (q, final_error_m).
    """
    q = HOME_QPOS.copy() if q0 is None else np.array(q0, dtype=float)
    for _ in range(iters):
        err = target - fk(q)
        if np.linalg.norm(err) < tol:
            break
        J = jacobian(q)
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(3), err)
        q = np.clip(q + dq, JOINT_LOWER, JOINT_UPPER)
    return q, float(np.linalg.norm(target - fk(q)))
