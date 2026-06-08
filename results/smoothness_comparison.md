# Smoothness Comparison: Quintic vs Torque MPC vs Position MPC

**Robot:** Fanuc CRX-25iA (6-DOF, 118kg)  
**Episodes:** 10 per controller  
**Date:** 2026-05-18

## Results

| Metric | Quintic+OL | Torque MPC | Position MPC | Torque MPC improvement |
|--------|:---------:|:---------:|:-----------:|:---------------------:|
| Accel RMS (rad/s²) | 4188 | **62** | 4267 | 68× smoother |
| Jerk RMS (rad/s³) | 1,048K | **11.4K** | 1,068K | 92× smoother |
| Ctrl Effort RMS | 0.69 | **0.28** | 0.76 | 2.5× less effort |
| Ctrl Rate RMS (1/s) | 165 | **5.1** | 171 | 32× smoother |
| EE Jerk RMS (m/s³) | 8583 | **793** | 8214 | 11× smoother |

Lower = smoother (except Ctrl Effort which is magnitude).

## Contact Rate (same N=10 run)

| Controller | Contact |
|-----------|:-------:|
| Quintic+OL | 3/10 |
| Torque MPC | 4/10 |
| Position MPC | 2/10 |

(N=10 is noisy; N=80 established rates are 40%, 38.8%, 36.2% respectively)

## Analysis

1. **Torque MPC is 1-2 orders of magnitude smoother** on every metric while achieving comparable contact rate.

2. **Quintic and Position MPC have nearly identical smoothness** because Position MPC outputs positions tracked by the same PD controller (kp=800). The PD amplifies every position error into sharp torque spikes regardless of how smooth the position trajectory is.

3. **Torque MPC bypasses PD entirely** — IPOPT directly optimizes torques subject to dynamics constraints. The smoothness cost (Δu penalty) naturally produces gentle accelerations.

4. **Control effort is 2.5× lower** — Torque MPC generates minimal torques to reach the target, while PD controllers saturate actuators fighting large position errors.

## Hardware Implications

- **Actuator longevity:** 92× less jerk means dramatically less mechanical stress on gears and bearings
- **Energy efficiency:** 2.5× less control effort = lower power consumption
- **Safety:** Smooth trajectories are more predictable for human coworkers (collaborative robot use case)
- **Same performance:** Contact rate is statistically equivalent (38.8% vs 40.0% at N=80)

## Methodology

Metrics computed from finite differences of logged joint trajectories:
- `ddq = diff(dq) / dt` (acceleration)
- `jerk = diff(ddq) / dt` (jerk)
- `ctrl_rate = diff(action) / dt` (torque rate)
- `ee_jerk = diff(diff(diff(ee_pos))) / dt³` (end-effector jerk)

Script: `compare_smoothness.py`
