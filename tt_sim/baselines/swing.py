"""Baseline swing generation."""

from tt_sim.interfaces import SwingPlanner, PaddleTarget, JointTrajectory
import numpy as np


class LerpSwingPlanner(SwingPlanner):
    """Stage 0 swing: linear interpolation from current joints to target joints.

    Uses an injectable IK function. If none is provided, uses a dummy
    (returns zeros) suitable for unit testing.
    """

    def __init__(self, n_joints: int = 7, n_steps: int = 50, ik_fn=None):
        self.n_joints = n_joints
        self.n_steps = n_steps
        self.ik_fn = ik_fn or (lambda target: np.zeros(n_joints))

    def plan(self, target: PaddleTarget, q_current: np.ndarray) -> JointTrajectory:
        q_target = self.ik_fn(target)
        # Make DOF-agnostic: match q_target length to q_current
        ndof = len(q_current)
        if len(q_target) > ndof:
            q_target = q_target[:ndof]
        elif len(q_target) < ndof:
            q_target = np.concatenate([q_target, np.zeros(ndof - len(q_target))])

        t_contact = max(target.t_contact, 0.01)  # avoid zero-duration
        times = np.linspace(0.0, t_contact, self.n_steps)
        alphas = np.linspace(0.0, 1.0, self.n_steps)
        positions = np.outer(1 - alphas, q_current) + np.outer(alphas, q_target)

        dt = times[1] - times[0] if self.n_steps > 1 else 1.0
        velocities = np.gradient(positions, dt, axis=0)

        return JointTrajectory(
            times=times,
            positions=positions,
            velocities=velocities,
        )


class MujocoIKSwingPlanner(SwingPlanner):
    """Stage 0 swing with MuJoCo Jacobian-based IK for the actual sim robot.

    Uses the MuJoCo model/data from the env to compute IK via the Jacobian
    transpose method, then LERP interpolates to the solution.
    """

    def __init__(self, env, ee_body: str = "EE", n_steps: int = 50, ik_iters: int = 100, n_joints: int = None):
        self.model = env.unwrapped.model
        self.data = env.unwrapped.data
        self.ee_body = ee_body
        self.n_steps = n_steps
        self.ik_iters = ik_iters
        self.n_joints = n_joints or self.model.nu  # auto-detect from actuator count

    def _ik(self, target_pos: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        """Damped least squares IK."""
        import mujoco

        nj = self.n_joints
        d = mujoco.MjData(self.model)
        d.qpos[:nj] = q_init.copy()
        mujoco.mj_forward(self.model, d)

        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body)
        damping = 0.01

        for _ in range(self.ik_iters):
            ee_pos = d.body(self.ee_body).xpos.copy()
            err = target_pos - ee_pos
            if np.linalg.norm(err) < 0.005:
                break

            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, d, jacp, None, ee_id)
            J = jacp[:, :nj]

            # Damped least squares: dq = J^T (J J^T + λ²I)^{-1} err
            JJT = J @ J.T + damping**2 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, err)
            d.qpos[:nj] += dq

            for i in range(nj):
                lo, hi = self.model.jnt_range[i]
                if lo != hi:
                    d.qpos[i] = np.clip(d.qpos[i], lo, hi)

            mujoco.mj_forward(self.model, d)

        return d.qpos[:nj].copy()

    def plan(self, target: PaddleTarget, q_current: np.ndarray) -> JointTrajectory:
        q_target = self._ik(target.position, q_current)

        t_contact = max(target.t_contact, 0.01)
        times = np.linspace(0.0, t_contact, self.n_steps)
        alphas = np.linspace(0.0, 1.0, self.n_steps)
        positions = np.outer(1 - alphas, q_current) + np.outer(alphas, q_target)

        dt = times[1] - times[0] if self.n_steps > 1 else 1.0
        velocities = np.gradient(positions, dt, axis=0)

        return JointTrajectory(
            times=times,
            positions=positions,
            velocities=velocities,
        )
