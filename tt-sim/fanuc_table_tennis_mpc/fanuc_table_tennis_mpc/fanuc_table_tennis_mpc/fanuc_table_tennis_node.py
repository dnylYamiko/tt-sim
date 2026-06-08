"""ROS 2 node for FANUC CRX-25iA table-tennis SINDy-MPC position control."""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Vector3Stamped
from builtin_interfaces.msg import Duration

from .ball_predictor import BallTrajectoryPredictor
from .intercept_planner import InterceptPlanner
from .crx25ia_kinematics import FANUCCRX25iA
from .position_mpc import SINDyPositionMPC
from .sindy_model import SINDYcModel


class FanucTableTennisMPCNode(Node):
    """
    Subscribes:
        /joint_states
        /ball_position
        /ball_velocity

    Publishes:
        /joint_trajectory_controller/joint_trajectory
    """

    def __init__(self):
        super().__init__("fanuc_table_tennis_mpc_node")
        self.dt = 0.02

        self.robot = FANUCCRX25iA()
        self.ball_predictor = BallTrajectoryPredictor(dt=self.dt)
        self.intercept_planner = InterceptPlanner()

        self.sindy = SINDYcModel()
        try:
            self.sindy.load("sindy_fanuc_crx25ia_model.npz")
            self.get_logger().info("Loaded trained SINDy model.")
        except Exception:
            self.sindy.use_fallback_model()
            self.get_logger().warn("No SINDy model found. Using fallback servo model.")

        self.mpc = SINDyPositionMPC(self.robot, self.sindy, dt=self.dt, horizon=10)

        self.joint_names = ["J1", "J2", "J3", "J4", "J5", "J6"]
        self.q_current = self.robot.q_home.copy()
        self.dq_current = np.zeros(6)
        self.ball_position = None
        self.ball_velocity = None

        self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 10)
        self.create_subscription(Vector3Stamped, "/ball_position", self.ball_position_callback, 10)
        self.create_subscription(Vector3Stamped, "/ball_velocity", self.ball_velocity_callback, 10)
        self.traj_pub = self.create_publisher(JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10)
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info("FANUC table-tennis MPC node started.")

    def joint_state_callback(self, msg):
        if len(msg.position) < 6:
            return
        name_to_pos = dict(zip(msg.name, msg.position))
        name_to_vel = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
        q, dq = [], []
        for name in self.joint_names:
            if name not in name_to_pos:
                return
            q.append(name_to_pos[name])
            dq.append(name_to_vel.get(name, 0.0))
        self.q_current = np.asarray(q, dtype=float)
        self.dq_current = np.asarray(dq, dtype=float)

    def ball_position_callback(self, msg):
        self.ball_position = np.array([msg.vector.x, msg.vector.y, msg.vector.z], dtype=float)

    def ball_velocity_callback(self, msg):
        self.ball_velocity = np.array([msg.vector.x, msg.vector.y, msg.vector.z], dtype=float)

    def control_loop(self):
        if self.ball_position is None or self.ball_velocity is None:
            return

        ball_state = np.hstack([self.ball_position, self.ball_velocity])
        ball_predictions = self.ball_predictor.predict(ball_state, horizon_steps=80)
        target_position, intercept_index = self.intercept_planner.choose_intercept(ball_predictions)

        x_current = np.hstack([self.q_current, self.dq_current])
        q_cmd, cost = self.mpc.solve(x_current, target_position)
        self.publish_joint_trajectory(q_cmd)

        ee = self.robot.end_effector_position(self.q_current)
        error = np.linalg.norm(ee - target_position)
        self.get_logger().info(
            f"target={target_position.round(3)} ee={ee.round(3)} error={error:.3f} cost={cost:.2f}"
        )

    def publish_joint_trajectory(self, q_cmd):
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = np.asarray(q_cmd, dtype=float).tolist()
        point.time_from_start = Duration(sec=0, nanosec=int(self.dt * 1e9))
        msg.points.append(point)
        self.traj_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FanucTableTennisMPCNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
