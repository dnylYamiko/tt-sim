"""ROS 2 fake ball publisher for testing the FANUC table-tennis MPC node."""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped


class FakeBallPublisher(Node):
    def __init__(self):
        super().__init__("fake_ball_publisher")
        self.dt = 0.02
        self.g = 9.81
        self.drag_coeff = 0.05
        self.initial_state = np.array([2.20, -0.35, 1.15, -1.45, 0.25, 0.35], dtype=float)
        self.state = self.initial_state.copy()

        self.pos_pub = self.create_publisher(Vector3Stamped, "/ball_position", 10)
        self.vel_pub = self.create_publisher(Vector3Stamped, "/ball_velocity", 10)
        self.timer = self.create_timer(self.dt, self.loop)

    def dynamics(self, state):
        v = state[3:]
        speed = np.linalg.norm(v) + 1e-9
        drag = -self.drag_coeff * speed * v
        a = np.array([drag[0], drag[1], drag[2] - self.g])
        return np.hstack([v, a])

    def step(self):
        dt = self.dt
        k1 = self.dynamics(self.state)
        k2 = self.dynamics(self.state + 0.5 * dt * k1)
        k3 = self.dynamics(self.state + 0.5 * dt * k2)
        k4 = self.dynamics(self.state + dt * k3)
        self.state = self.state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        if self.state[2] <= 0.0:
            self.state = self.initial_state.copy()

    def loop(self):
        now = self.get_clock().now().to_msg()
        pos_msg = Vector3Stamped()
        vel_msg = Vector3Stamped()
        pos_msg.header.stamp = now
        vel_msg.header.stamp = now
        pos_msg.header.frame_id = "base_link"
        vel_msg.header.frame_id = "base_link"

        pos_msg.vector.x = float(self.state[0])
        pos_msg.vector.y = float(self.state[1])
        pos_msg.vector.z = float(self.state[2])
        vel_msg.vector.x = float(self.state[3])
        vel_msg.vector.y = float(self.state[4])
        vel_msg.vector.z = float(self.state[5])

        self.pos_pub.publish(pos_msg)
        self.vel_pub.publish(vel_msg)
        self.step()


def main(args=None):
    rclpy.init(args=args)
    node = FakeBallPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
