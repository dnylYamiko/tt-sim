"""Symbolic RNEA dynamics from MuJoCo model for CasADi NLP.

Builds symbolic inverse-dynamics functions:
    tau = RNEA(q, qd, qdd)
    g(q) = RNEA(q, 0, 0)
    M(q) via column-by-column RNEA
    C(q, qd)*qd = RNEA(q, qd, 0) - g(q)
"""

from __future__ import annotations

import casadi as ca
import mujoco
import numpy as np

from tt_sim.upgrades.mpc_fk import _rotation_matrix


def _quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix (NumPy constant)."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def _skew(v: ca.SX) -> ca.SX:
    """3D skew-symmetric matrix from vector."""
    return ca.vertcat(
        ca.horzcat(0, -v[2], v[1]),
        ca.horzcat(v[2], 0, -v[0]),
        ca.horzcat(-v[1], v[0], 0),
    )


def build_dynamics_casadi(model: mujoco.MjModel, n_joints: int) -> dict:
    """Build symbolic RNEA from MuJoCo model.

    Returns dict with:
        'rnea_fn':         CasADi Function (q, qd, qdd) -> tau
        'gravity_fn':      CasADi Function (q,) -> tau_gravity
        'mass_matrix_fn':  CasADi Function (q,) -> M (n_joints x n_joints)
        'coriolis_fn':     CasADi Function (q, qd) -> C(q,qd)*qd
        'fk_fn':           CasADi Function q -> p_ee (reuse)
        'fk_rot_fn':       CasADi Function q -> (p_ee, R_flat)
        'joint_ranges':    (n_joints, 2) numpy array
        'n_joints':        int
        'joint_damping':   (n_joints,) numpy array
        'gear_ratios':     (n_joints,) numpy array
    """
    # ── Extract kinematic chain ──────────────────────────────────────────
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "EE")
    chain = []
    body_id = ee_id
    while body_id > 0:
        chain.append(body_id)
        body_id = model.body_parentid[body_id]
    chain.reverse()  # world -> ... -> EE

    # Collect link parameters
    links = []  # list of dicts per link in chain
    joint_ranges = np.zeros((n_joints, 2))
    joint_damping = np.zeros(n_joints)
    gear_ratios = np.zeros(n_joints)

    joint_idx = 0
    for body_id in chain:
        link = {
            'body_id': body_id,
            'pos': model.body_pos[body_id].copy(),        # offset from parent
            'quat': model.body_quat[body_id].copy(),      # (w,x,y,z)
            'mass': float(model.body_mass[body_id]),
            'com': model.body_ipos[body_id].copy(),        # CoM in body frame
            'inertia_diag': model.body_inertia[body_id].copy(),  # principal moments
            'inertia_quat': model.body_iquat[body_id].copy(),    # principal frame orient
            'joint_axis': None,
            'joint_idx': None,
        }

        # Find joint for this body
        for jnt_id in range(model.njnt):
            if model.jnt_bodyid[jnt_id] == body_id and joint_idx < n_joints:
                link['joint_axis'] = model.jnt_axis[jnt_id].copy()
                link['joint_idx'] = joint_idx
                joint_ranges[joint_idx] = model.jnt_range[jnt_id].copy()

                # Joint damping
                dof_id = model.jnt_dofadr[jnt_id]
                joint_damping[joint_idx] = float(model.dof_damping[dof_id])

                joint_idx += 1
                break

        links.append(link)

    # Extract gear ratios from actuators
    for act_id in range(model.nu):
        if act_id < n_joints:
            gear_ratios[act_id] = float(model.actuator_gear[act_id, 0])

    # Gravity
    gravity = model.opt.gravity.copy()  # (3,) typically [0, 0, -9.81]

    # ── Symbolic variables ───────────────────────────────────────────────
    q = ca.SX.sym('q', n_joints)
    qd = ca.SX.sym('qd', n_joints)
    qdd = ca.SX.sym('qdd', n_joints)

    # ── Forward pass: compute transforms, velocities, accelerations ─────
    # For each link: R_i (world->link), p_i (link origin in world),
    #                omega_i (angular vel in world), v_i (linear vel of origin in world),
    #                alpha_i (angular acc), a_i (linear acc of origin)

    n_links = len(links)

    # World frame initial conditions
    R_world = ca.SX.eye(3)
    p_world = ca.SX.zeros(3)
    omega_world = ca.SX.zeros(3)
    v_world = ca.SX.zeros(3)
    alpha_world = ca.SX.zeros(3)
    # Linear acceleration of world frame origin includes gravity effect
    # In RNEA, we set a_0 = -gravity so that gravity appears naturally in the
    # backward pass without adding it separately.
    a_world = -ca.SX(gravity)

    # Storage for forward pass results
    R_list = [R_world]     # R_list[0] = world
    p_list = [p_world]
    omega_list = [omega_world]
    v_list = [v_world]
    alpha_list = [alpha_world]
    a_list = [a_world]

    for i, link in enumerate(links):
        # Parent transform
        R_parent = R_list[-1]
        p_parent = p_list[-1]
        omega_parent = omega_list[-1]
        v_parent = v_list[-1]
        alpha_parent = alpha_list[-1]
        a_parent = a_list[-1]

        # Body offset from parent (constant)
        pos_const = link['pos']
        quat_const = link['quat']
        R_body_const = _quat_to_rotmat(quat_const)

        # Apply body constant transform
        R_i = R_parent @ R_body_const
        p_i = p_parent + R_parent @ ca.SX(pos_const)

        # Apply joint rotation if this link has a joint
        if link['joint_idx'] is not None:
            ji = link['joint_idx']
            axis = link['joint_axis']
            R_jnt = _rotation_matrix(axis, q[ji])
            R_i = R_i @ R_jnt

            # Joint axis in world frame
            z_i = R_i @ ca.SX(axis)  # but axis is in local pre-joint frame...
            # Actually: axis is defined in the body frame (after body transform,
            # before joint rotation). The joint rotation rotates about this axis.
            # In world frame, the axis is R_parent @ R_body_const @ axis
            # (before joint rotation is applied, but the axis direction is the same
            # after rotation since we rotate about it).
            z_i = R_parent @ R_body_const @ ca.SX(axis)

            # Angular velocity
            omega_i = omega_parent + z_i * qd[ji]

            # Angular acceleration
            alpha_i = alpha_parent + z_i * qdd[ji] + _skew(omega_parent) @ z_i * qd[ji]

            # Linear velocity of link origin
            r_parent_to_i = p_i - p_parent
            v_i = v_parent + _skew(omega_parent) @ r_parent_to_i

            # Linear acceleration of link origin
            a_i = (a_parent
                   + _skew(alpha_parent) @ r_parent_to_i
                   + _skew(omega_parent) @ (_skew(omega_parent) @ r_parent_to_i))
        else:
            omega_i = omega_parent
            alpha_i = alpha_parent
            r_parent_to_i = p_i - p_parent
            v_i = v_parent + _skew(omega_parent) @ r_parent_to_i
            a_i = (a_parent
                   + _skew(alpha_parent) @ r_parent_to_i
                   + _skew(omega_parent) @ (_skew(omega_parent) @ r_parent_to_i))

        R_list.append(R_i)
        p_list.append(p_i)
        omega_list.append(omega_i)
        v_list.append(v_i)
        alpha_list.append(alpha_i)
        a_list.append(a_i)

    # ── Backward pass: compute forces and torques ────────────────────────
    # For each link (tip to base), compute the net force and torque at
    # the joint, then project onto joint axis to get tau_i.

    # f_list[i] = force exerted ON link i by its parent (in world frame)
    # n_list[i] = torque exerted ON link i by its parent (in world frame)

    # Initialize: force/torque from "child" of last link = 0
    f_tip = ca.SX.zeros(3)
    n_tip = ca.SX.zeros(3)

    tau = ca.SX.zeros(n_joints)

    # We process links in reverse. f_child, n_child accumulate from children.
    f_child = f_tip
    n_child = n_tip

    for i in range(n_links - 1, -1, -1):
        link = links[i]
        # Index into R_list/p_list is i+1 (since index 0 = world)
        R_i = R_list[i + 1]
        p_i = p_list[i + 1]
        omega_i = omega_list[i + 1]
        alpha_i = alpha_list[i + 1]
        a_i = a_list[i + 1]

        m = link['mass']

        # CoM position in world frame
        com_local = ca.SX(link['com'])
        r_com = R_i @ com_local  # CoM offset from link origin, in world frame
        p_com = p_i + r_com

        # Inertia tensor in world frame
        # Principal inertia is diagonal in principal frame.
        # Principal frame orientation: body_iquat
        I_diag = link['inertia_diag']
        I_quat = link['inertia_quat']
        R_inertia = _quat_to_rotmat(I_quat)  # principal -> body frame
        I_principal = np.diag(I_diag)
        # Inertia in body frame: R_inertia @ I_principal @ R_inertia.T
        I_body = R_inertia @ I_principal @ R_inertia.T
        # Inertia in world frame: R_i @ I_body @ R_i.T
        I_world = R_i @ ca.SX(I_body) @ R_i.T

        # Acceleration of CoM
        a_com = a_i + _skew(alpha_i) @ r_com + _skew(omega_i) @ (_skew(omega_i) @ r_com)

        # Newton: F_i = m * a_com (includes gravity via a_world = -g trick)
        F_i = m * a_com

        # Euler: N_i = I * alpha + omega x (I * omega) about CoM in world frame
        N_i = I_world @ alpha_i + _skew(omega_i) @ (I_world @ omega_i)

        # Force balance at link origin:
        # f_i (from parent) = F_i + f_child (passed to child link)
        f_i = F_i + f_child

        # Torque balance about link origin:
        # n_i = N_i + r_com x F_i + n_child + r_child x f_child
        # where r_child is from this link's origin to child link's origin
        if i < n_links - 1:
            p_child = p_list[i + 2]  # child link origin
            r_to_child = p_child - p_i
            n_i = N_i + _skew(r_com) @ F_i + n_child + _skew(r_to_child) @ f_child
        else:
            n_i = N_i + _skew(r_com) @ F_i + n_child

        # Project torque onto joint axis
        if link['joint_idx'] is not None:
            ji = link['joint_idx']
            axis = link['joint_axis']
            # Joint axis in world frame (same as forward pass)
            R_parent = R_list[i]  # parent is at index i in R_list
            R_body_const = _quat_to_rotmat(link['quat'])
            z_i = R_parent @ R_body_const @ ca.SX(axis)
            tau[ji] = ca.dot(n_i, z_i)

        # Pass force/torque up to parent
        f_child = f_i
        n_child = n_i + _skew(p_i - p_list[i]) @ f_i  # torque about parent origin
        # Actually, the backward recursion should pass f_i and n_i about THIS link's
        # origin up. The parent will account for the moment arm.
        # Let me reconsider: n_child for the parent = n_i (about link i origin)
        # The parent computes: n_parent includes n_child + r_to_child x f_child
        # where r_to_child = p_i - p_parent.
        # So we should pass f_child = f_i, n_child = n_i (about link i origin).
        # But n_i above is computed about link i origin already. Wait, let me
        # re-derive more carefully.

        # Actually the standard RNEA backward pass works as follows:
        # f_i = m_i * a_c_i - f_{i+1} ... no, let me use the standard formulation.
        # I'll redo this more carefully.
        f_child = f_i
        n_child = n_i

    # ── Add joint damping to torques ─────────────────────────────────────
    for ji in range(n_joints):
        tau[ji] = tau[ji] + joint_damping[ji] * qd[ji]

    # ── Build CasADi functions ───────────────────────────────────────────
    rnea_fn = ca.Function('rnea', [q, qd, qdd], [tau], ['q', 'qd', 'qdd'], ['tau'])

    # Gravity: rnea(q, 0, 0)
    g_tau = rnea_fn(q, ca.SX.zeros(n_joints), ca.SX.zeros(n_joints))
    gravity_fn = ca.Function('gravity', [q], [g_tau], ['q'], ['tau'])

    # Coriolis: rnea(q, qd, 0) - g(q)
    c_tau = rnea_fn(q, qd, ca.SX.zeros(n_joints)) - gravity_fn(q)
    coriolis_fn = ca.Function('coriolis', [q, qd], [c_tau], ['q', 'qd'], ['tau'])

    # Mass matrix: column by column
    # M(:, i) = rnea(q, 0, e_i) - g(q)
    M_cols = []
    for ji in range(n_joints):
        e_i = ca.SX.zeros(n_joints)
        e_i[ji] = 1.0
        col = rnea_fn(q, ca.SX.zeros(n_joints), e_i) - gravity_fn(q)
        M_cols.append(col)
    M = ca.horzcat(*M_cols)
    mass_matrix_fn = ca.Function('mass_matrix', [q], [M], ['q'], ['M'])

    # ── Also build FK (reuse from mpc_fk pattern) ───────────────────────
    # EE position and rotation from the forward pass
    p_ee = p_list[-1]  # last link (EE)
    R_ee = R_list[-1]
    fk_fn = ca.Function('fk_pos', [q], [p_ee], ['q'], ['p_ee'])
    R_flat = ca.vertcat(R_ee[0, :].T, R_ee[1, :].T, R_ee[2, :].T)
    fk_rot_fn = ca.Function('fk_rot', [q], [p_ee, R_flat], ['q'], ['p_ee', 'R_flat'])

    return {
        'rnea_fn': rnea_fn,
        'gravity_fn': gravity_fn,
        'mass_matrix_fn': mass_matrix_fn,
        'coriolis_fn': coriolis_fn,
        'fk_fn': fk_fn,
        'fk_rot_fn': fk_rot_fn,
        'joint_ranges': joint_ranges,
        'n_joints': n_joints,
        'joint_damping': joint_damping,
        'gear_ratios': gear_ratios,
    }
