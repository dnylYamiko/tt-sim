# Fanuc CRX-25iA — Stage-by-Stage Upgrade Progression

**Date:** 2026-05-17
**Robot:** Fanuc CRX-25iA (6-DOF, 118kg, gear=[500,500,300,150,80,50])
**Episodes per stage:** 40
**Sim:** fancy_gym TableTennis4D-v0, dt=0.008s (125Hz), custom floor-mounted XML
**PD gains:** kp_scale=800, kd_scale=50, gravity_comp=True, robot_x=1.0

---

## Summary Table

| # | Stage | Predictor | Aimer | Swing | Control | Contact % | Landing % | Mean Reward |
|---|-------|-----------|-------|-------|---------|-----------|-----------|-------------|
| 0 | Baseline (all Stage-0) | ballistic | face_net | lerp | open_loop | 2.5% | 0.0% | 0.218 |
| 1 | + Bounce predictor | **drag_bounce** | face_net | lerp | open_loop | 7.5% | 0.0% | 0.329 |
| 2 | + Specular aimer | drag_bounce | **specular** | lerp | open_loop | 15.0% | 0.0% | 0.486 |
| 3 | + Quintic swing | drag_bounce | specular | **quintic** | open_loop | **57.5%** | **7.5%** | **1.714** |
| 4 | + Replan control | drag_bounce | specular | quintic | **replan** | 17.5% | 2.5% | 0.756 |
| 5 | Position MPC (2a) | drag_bounce | specular | quintic | **mpc** | 27.5% | 0.0% | 0.903 |
| 6 | Torque MPC (2b) | drag_bounce | face_net | lerp | **torque_mpc** | 20.0% | 0.0% | 0.628 |

---

## Stage 0: All Baselines

All subsystems at Stage 0. Ballistic predictor (no drag, no bounce), face-the-net
aimer, linear interpolation swing, plan-once open-loop control.

**Why this fails:** The ballistic predictor extrapolates a simple parabola. The ball
bounces off the table at t≈0.5s, but the predictor doesn't model this — it predicts
the ball at z<0 (through the table) at intercept time. The z-coordinate gets clipped
to table height (0.76m), so the paddle targets z=0.76 while the actual ball is at
z≈1.4 after bounce. This is the dominant failure mode.

```bash
# Stage 0: All baselines
python run.py --robot fanuc \
  --predictor ballistic --aimer face_net --swing lerp --control open_loop \
  --episodes 40 --log results/fanuc_s0_baseline.csv
```

| Metric | Value |
|--------|-------|
| Contact rate | 2.5% |
| Landing rate | 0.0% |
| Mean reward | 0.218 |

---

## Stage 1: Upgrade Predictor (ballistic → drag_bounce)

**What changed:** The DragBouncePredictor models gravity, aerodynamic drag (c_d=0.0
in MuJoCo), AND table bounce with coefficient of restitution (e=0.91). This correctly
predicts the ball's z-coordinate after bouncing off the table.

**Why upgrade this first:** Without bounce prediction, no downstream controller can aim
at the correct height. The predictor is the foundation of the entire planning pipeline.

**Result:** Contact 2.5% → 7.5% (+5pp). The target position is now at the correct
height (z≈1.4 instead of z=0.76). The improvement is modest because the face_net
aimer and lerp swing still limit performance.

```bash
# Stage 1: + drag_bounce predictor
python run.py --robot fanuc \
  --predictor drag_bounce --aimer face_net --swing lerp --control open_loop \
  --episodes 40 --log results/fanuc_s1_dragbounce.csv
```

| Metric | Value |
|--------|-------|
| Contact rate | 7.5% |
| Landing rate | 0.0% |
| Mean reward | 0.329 |

---

## Stage 2: Upgrade Aimer (face_net → specular)

**What changed:** FaceNetAimer points the paddle normal toward the net center — a fixed
direction regardless of incoming ball angle. SpecularAimer computes the specular
reflection: given the incoming ball velocity and desired landing target, it solves for
the paddle normal that reflects the ball toward the opponent's side.

**Why upgrade this second:** With correct prediction (Stage 1), we now hit the right
zone but the paddle orientation is arbitrary. Specular aiming computes a physically
meaningful paddle normal, which also affects the IK solution for joint configuration —
a better target pose gives a better trajectory.

**Result:** Contact 7.5% → 15.0% (+7.5pp). The specular aimer produces a more
physically consistent target that the IK solver can reach more reliably.

```bash
# Stage 2: + specular aimer
python run.py --robot fanuc \
  --predictor drag_bounce --aimer specular --swing lerp --control open_loop \
  --episodes 40 --log results/fanuc_s2_specular.csv
```

| Metric | Value |
|--------|-------|
| Contact rate | 15.0% |
| Landing rate | 0.0% |
| Mean reward | 0.486 |

---

## Stage 3: Upgrade Swing (lerp → quintic)

**What changed:** LerpSwingPlanner does linear interpolation in joint space from the
current config to the IK-solved target. QuinticSwingPlanner uses 5th-order polynomials
with zero initial/final velocity and acceleration, yielding smooth, dynamically
feasible trajectories. It also uses the Jacobian for Cartesian velocity mapping at
the contact point.

**Why this is the biggest single improvement:** The quintic polynomial produces a
velocity profile that respects physical constraints — zero velocity at start (from
rest) and a controlled arrival velocity at contact. The Fanuc's aggressive PD gains
(kp_scale=800) can track quintic trajectories much better than lerp's constant-velocity
profile, which causes actuator saturation at trajectory start/end. The smooth
acceleration profile keeps all actuators within their ctrl limits.

**Result:** Contact 15.0% → **57.5%** (+42.5pp), Landing 0% → **7.5%**. This is the
largest single improvement in the entire pipeline. The quintic swing is the critical
enabler for the Fanuc's high-gain PD controller.

```bash
# Stage 3: + quintic swing
python run.py --robot fanuc \
  --predictor drag_bounce --aimer specular --swing quintic --control open_loop \
  --episodes 40 --log results/fanuc_s3_quintic.csv
```

| Metric | Value |
|--------|-------|
| Contact rate | **57.5%** |
| Landing rate | **7.5%** |
| Mean reward | **1.714** |

---

## Stage 4: Upgrade Control (open_loop → replan)

**What changed:** OpenLoopController plans once after 8 observations (~64ms) and
executes the trajectory without correction. ReplanController re-solves the aiming
and swing planning every few frames as new ball observations arrive.

**Why this hurts on Fanuc:** Replanning causes the target joint configuration to shift
every few frames. Each replan generates a new quintic trajectory from the *current*
joint state to the *updated* target. On the heavy Fanuc arm (118kg), these frequent
trajectory switches cause the PD controller to chase a moving target — the arm never
fully commits to one swing. The open-loop approach works better here because it
commits to a single, smooth trajectory and lets the aggressive PD gains track it
faithfully.

**Result:** Contact 57.5% → 17.5% (−40pp). **Replanning hurts on Fanuc.** The heavy
arm benefits from commitment to a single trajectory rather than frequent re-planning.
This is consistent with the Fanuc's high inertia — trajectory switches waste energy
on deceleration/re-acceleration.

```bash
# Stage 4: + replan control
python run.py --robot fanuc \
  --predictor drag_bounce --aimer specular --swing quintic --control replan \
  --episodes 40 --log results/fanuc_s4_replan.csv
```

| Metric | Value |
|--------|-------|
| Contact rate | 17.5% |
| Landing rate | 2.5% |
| Mean reward | 0.756 |

---

## Stage 5: Position MPC (Stage 2a)

**What changed:** Replaced the swing+control pipeline with a CasADi nonlinear program
(NLP). The MPC directly optimizes joint positions over a receding horizon to minimize
distance to the predicted contact point, subject to joint limits. Uses symbolic forward
kinematics from the MuJoCo XML, IPOPT solver with warm-starting between replans.

**Why partial recovery:** The MPC also replans every few frames (like Stage 4), but
its warm-started NLP produces smoother trajectory updates than the discrete
replan-from-scratch approach. The continuous optimization landscape avoids the sharp
trajectory discontinuities that hurt Stage 4. However, it still doesn't match the
open-loop quintic's committed swing.

**Result:** Contact 17.5% → 27.5% (+10pp). Better than naive replanning but still
below the open-loop quintic peak. The MPC is still fighting the Fanuc's inertia with
frequent replans.

```bash
# Stage 5: Position MPC
python run.py --robot fanuc \
  --predictor drag_bounce --aimer specular --swing quintic --control mpc \
  --episodes 40 --log results/fanuc_s5_mpc.csv
```

| Metric | Value |
|--------|-------|
| Contact rate | 27.5% |
| Landing rate | 0.0% |
| Mean reward | 0.903 |

---

## Stage 6: Torque MPC (Stage 2b)

**What changed:** The torque MPC bypasses the PD controller entirely. It outputs raw
ctrl-space torques computed by a CasADi NLP with full rigid-body dynamics constraints
via symbolic RNEA (Recursive Newton-Euler Algorithm). The dynamics model (mass matrix
M, Coriolis C, gravity g) is built symbolically from the MuJoCo XML and matches
MuJoCo's forward dynamics exactly (verified: 0.0 error).

**Key design choices:**
- `w_ori=0`: Orientation cost disabled — position and orientation fight each other
  on a 6-DOF arm without redundancy.
- Uses face_net aimer + lerp swing (not specular + quintic) because the NLP handles
  trajectory optimization internally.
- Dynamics constraints (M*qdd + C*qd + g = tau) ensure physical realizability.

**Why lower than expected on Fanuc:** The torque MPC was designed to bypass the PD
bottleneck. On WAM (14kg, 7-DOF), this helps because PD tracking is weak. But Fanuc
has aggressive PD gains (kp_scale=800 vs WAM's 50) and gravity compensation, so PD
tracking is already good. The torque MPC's advantage — dynamics-feasible trajectories
— is less impactful when the PD controller already tracks well. Additionally, the
NLP's effort penalty and dynamics constraints limit how aggressively the arm can swing,
while the PD+quintic approach just "commits and pushes."

**Result:** Contact 20.0%, Landing 0.0%. On Fanuc, the open-loop quintic (Stage 3) at
57.5% contact remains the best approach. The torque MPC's architecture is more
principled but the Fanuc's strong PD controller makes it unnecessary for this task.

```bash
# Stage 6: Torque MPC
python run.py --robot fanuc \
  --predictor drag_bounce --aimer face_net --swing lerp --control torque_mpc \
  --episodes 40 --log results/fanuc_s6_torque_mpc.csv
```

| Metric | Value |
|--------|-------|
| Contact rate | 20.0% |
| Landing rate | 0.0% |
| Mean reward | 0.628 |

---

## Key Findings

1. **Quintic swing is the most impactful upgrade on Fanuc** (+42.5pp contact). The
   smooth 5th-order polynomial trajectory is perfectly suited for the Fanuc's
   high-gain PD controller with gravity compensation. This is the single biggest
   improvement in the entire pipeline.

2. **Predictor must model bounces.** Without the drag_bounce predictor, all controllers
   target z=0.76 (table height) instead of z≈1.4 (post-bounce height). This is a
   necessary but not sufficient condition for good performance.

3. **Replanning hurts on heavy arms.** The Fanuc (118kg) benefits from committing to a
   single smooth trajectory rather than frequent re-planning. Each replan causes the
   arm to partially decelerate and re-accelerate, wasting energy and time. The
   open-loop approach (Stage 3) outperforms replan (Stage 4) by 40pp.

4. **The optimal Fanuc configuration is drag_bounce + specular + quintic + open_loop**
   at 57.5% contact and 7.5% landing — all Stage-1 subsystems with no MPC overhead.

5. **Torque MPC's advantage is architecture-dependent.** On the lightweight WAM (14kg)
   with weak PD gains, torque MPC helps by bypassing the PD bottleneck. On the heavy
   Fanuc with strong PD + gravity comp, the bottleneck doesn't exist — the PD already
   tracks well. The torque MPC adds NLP solve overhead without commensurate benefit.

6. **Position MPC partially recovers from the replanning penalty** (+10pp over naive
   replan) because warm-started NLP produces smoother trajectory updates. But it
   still can't match the committed open-loop quintic.

---

## Failed Experiments

### Running Orientation + Velocity Ramp Costs (S5/S6)

**Hypothesis:** Adding linearly ramped orientation and velocity costs during the second
half of the MPC horizon would help the optimizer "prepare" the paddle alignment and
swing speed before contact, without fighting the position cost early on.

**Changes tested:** For Position MPC: added running ori ramp (0.1*alpha*w_ori) and
velocity ramp (0.1*alpha*w_vel) for k >= N//2, bumped w_vel 1.0→10.0. For Torque MPC:
added running ori ramp (w_ori_run=10) and velocity ramp (0.1*alpha_v*w_vel) for
k >= N//2, increased horizon 10→15, max_iter 100→150.

**Result:** Both got worse. Position MPC: 27.5% → 15.0% contact. Torque MPC: 20.0% →
12.5% contact. The additional running costs over-constrain the optimizer mid-trajectory,
degrading position tracking which is the primary determinant of contact on Fanuc.

**Conclusion:** On Fanuc, terminal-only costs work best. The arm needs all its
optimization budget focused on reaching the right position; adding mid-trajectory
orientation/velocity costs diverts effort from this critical objective. **Reverted.**

---

## Reproduction

All scripts assume the working directory is `tt-sim/` with the venv activated:

```bash
cd /path/to/tt-sim
source .venv/bin/activate
```

Run all stages and parse results:

```bash
# Run all 7 stages sequentially
python run.py --robot fanuc --predictor ballistic  --aimer face_net  --swing lerp    --control open_loop  --episodes 40 --log results/fanuc_s0_baseline.csv
python run.py --robot fanuc --predictor drag_bounce --aimer face_net  --swing lerp    --control open_loop  --episodes 40 --log results/fanuc_s1_dragbounce.csv
python run.py --robot fanuc --predictor drag_bounce --aimer specular  --swing lerp    --control open_loop  --episodes 40 --log results/fanuc_s2_specular.csv
python run.py --robot fanuc --predictor drag_bounce --aimer specular  --swing quintic --control open_loop  --episodes 40 --log results/fanuc_s3_quintic.csv
python run.py --robot fanuc --predictor drag_bounce --aimer specular  --swing quintic --control replan     --episodes 40 --log results/fanuc_s4_replan.csv
python run.py --robot fanuc --predictor drag_bounce --aimer specular  --swing quintic --control mpc        --episodes 40 --log results/fanuc_s5_mpc.csv
python run.py --robot fanuc --predictor drag_bounce --aimer face_net  --swing lerp    --control torque_mpc --episodes 40 --log results/fanuc_s6_torque_mpc.csv

# Parse all results
python -c "
import csv, glob
files = sorted(glob.glob('results/fanuc_s*.csv'))
print(f'| File | Contact | Landing | Reward |')
print(f'|------|---------|---------|--------|')
for f in files:
    rows = list(csv.DictReader(open(f)))
    n = len(rows)
    c = sum(1 for r in rows if r['contact'] == 'True')
    l = sum(1 for r in rows if r['landed'] == 'True')
    rw = sum(float(r['reward']) for r in rows) / n
    print(f'| {f.split(\"/\")[-1]:<35} | {c/n*100:5.1f}% | {l/n*100:5.1f}% | {rw:.3f} |')
"
```
