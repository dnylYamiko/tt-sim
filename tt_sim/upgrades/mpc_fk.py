"""Symbolic forward kinematics from MuJoCo model for CasADi NLP."""

from __future__ import annotations

import casadi as ca
import mujoco
import numpy as np


def _rotation_matrix(axis: np.ndarray, theta: ca.SX) -> ca.SX:
    """Rodrigues rotation: 3x3 rotation matrix for angle theta about unit axis."""
    ax = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = float(ax[0]), float(ax[1]), float(ax[2])
    c = ca.cos(theta)
    s = ca.sin(theta)
    C = 1 - c
    return ca.vertcat(
        ca.horzcat(x*x*C + c,   x*y*C - z*s, x*z*C + y*s),
        ca.horzcat(y*x*C + z*s, y*y*C + c,   y*z*C - x*s),
        ca.horzcat(z*x*C - y*s, z*y*C + x*s, z*z*C + c),
    )


def build_fk_casadi(model: mujoco.MjModel, n_joints: int) -> dict:
    """Build symbolic FK from MuJoCo model.

    Returns dict with:
        'q': CasADi SX symbol (n_joints,)
        'p_ee': CasADi SX expression (3,) — EE position in world frame
        'R_ee': CasADi SX expression (3,3) — EE rotation in world frame
        'fk_fn': CasADi Function  q -> p_ee
        'fk_rot_fn': CasADi Function  q -> (p_ee, R_ee_flat)
        'joint_ranges': (n_joints, 2) numpy array of (lo, hi)
    """
    # Extract kinematic chain from MuJoCo model
    # Walk the body tree from worldbody to EE, collecting joint axes and body offsets
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "EE")

    # Build chain from EE back to world
    chain = []
    body_id = ee_id
    while body_id > 0:
        chain.append(body_id)
        body_id = model.body_parentid[body_id]
    chain.reverse()  # world -> ... -> EE

    # For each body in chain, get:
    #   - body offset from parent: model.body_pos[body_id]
    #   - body quaternion from parent: model.body_quat[body_id]
    #   - joint (if any): axis, range

    q = ca.SX.sym('q', n_joints)
    joint_ranges = np.zeros((n_joints, 2))

    # Start with identity transform
    R = ca.SX.eye(3)
    p = ca.SX.zeros(3)

    joint_idx = 0
    for body_id in chain:
        # Body frame offset from parent
        pos = model.body_pos[body_id].copy()
        quat = model.body_quat[body_id].copy()  # (w, x, y, z)

        # Convert body quaternion to rotation matrix (constant)
        w, x, y, z = quat
        body_R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
            [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
        ])

        # Apply body transform: p = p + R @ pos, R = R @ body_R
        p = p + R @ pos
        if not np.allclose(quat, [1, 0, 0, 0]):
            R = R @ body_R

        # Check if this body has a joint
        # Find joints belonging to this body
        for jnt_id in range(model.njnt):
            if model.jnt_bodyid[jnt_id] == body_id and joint_idx < n_joints:
                axis = model.jnt_axis[jnt_id].copy()
                joint_ranges[joint_idx] = model.jnt_range[jnt_id].copy()

                # Apply joint rotation
                R_jnt = _rotation_matrix(axis, q[joint_idx])
                R = R @ R_jnt
                joint_idx += 1

    p_ee = p
    R_ee = R

    fk_fn = ca.Function('fk_pos', [q], [p_ee], ['q'], ['p_ee'])
    # Flatten row-major to match NumPy convention (CasADi reshape is col-major)
    R_flat = ca.vertcat(R_ee[0, :].T, R_ee[1, :].T, R_ee[2, :].T)
    fk_rot_fn = ca.Function('fk_rot', [q], [p_ee, R_flat], ['q'], ['p_ee', 'R_flat'])

    return {
        'q': q,
        'p_ee': p_ee,
        'R_ee': R_ee,
        'fk_fn': fk_fn,
        'fk_rot_fn': fk_rot_fn,
        'joint_ranges': joint_ranges,
    }
