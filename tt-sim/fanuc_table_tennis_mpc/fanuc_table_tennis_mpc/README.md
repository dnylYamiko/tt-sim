# FANUC CRX-25iA Table-Tennis SINDy-MPC

This project contains a FANUC CRX-25iA-style table-tennis interception controller using position-level MPC and a SINDy/SINDYc prediction model.

## Contents

- `crx25ia_kinematics.py` - approximate CRX-25iA kinematics
- `ball_predictor.py` - table-tennis ball trajectory predictor
- `intercept_planner.py` - interception target selector
- `sindy_model.py` - SINDy/SINDYc model implementation
- `position_mpc.py` - baseline MPC and SINDy-MPC controllers
- `tt_sim_adapter.py` - adapter for `tt_sim` BallObservation objects
- `offline_simulation.py` - standalone simulation and comparison plots
- `fanuc_table_tennis_node.py` - ROS 2 MPC control node
- `fake_ball_publisher.py` - ROS 2 fake ball publisher for testing

## Offline run

```bash
pip install -r requirements.txt
python -m fanuc_table_tennis_mpc.offline_simulation
```

## ROS 2 run

Copy this folder into `~/ros2_ws/src`, then:

```bash
cd ~/ros2_ws
colcon build --packages-select fanuc_table_tennis_mpc
source install/setup.bash
ros2 launch fanuc_table_tennis_mpc table_tennis_mpc.launch.py
```

The ROS 2 node publishes joint-position commands to:

```text
/joint_trajectory_controller/joint_trajectory
```

## Safety

The included FANUC model is approximate and for academic simulation only. Use official FANUC/ROBOGUIDE/URDF data and safety-limited testing before connecting to a real robot.

## Performance evaluation plots

The project now includes `fanuc_table_tennis_mpc/evaluation.py`, which computes and saves controller performance metrics and plots.

Run:

```bash
python -m fanuc_table_tennis_mpc.offline_simulation
```

The script compares baseline position-MPC against SINDy-MPC and writes outputs to:

```text
results/plots/
```

Generated evaluation outputs include:

- `performance_metrics.csv`
- `01_tracking_error_vs_time.png`
- `02_xyz_component_errors.png`
- `03_sindy_end_effector_vs_target.png`
- `04_mpc_cost_vs_time.png`
- `05_sindy_joint_positions.png`
- `06_sindy_joint_velocities.png`
- `07_sindy_joint_position_commands.png`
- `08_command_rate_norm.png`
- `09_summary_metric_bars.png`
- `10_3d_trajectories.png`
- `11_final_robot_pose.png`

Metrics include RMSE, MAE, mean error, final error, maximum error, success rate, time-to-threshold, mean joint speed, command smoothness and MPC cost.
