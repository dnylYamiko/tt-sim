from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="fanuc_table_tennis_mpc",
            executable="fake_ball_publisher",
            name="fake_ball_publisher",
            output="screen",
        ),
        Node(
            package="fanuc_table_tennis_mpc",
            executable="fanuc_table_tennis_mpc_node",
            name="fanuc_table_tennis_mpc_node",
            output="screen",
        ),
    ])
