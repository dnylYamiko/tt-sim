#!/usr/bin/env python3
"""Verify CRX-25iA URDF loads in MuJoCo and FK matches ground truth."""
import numpy as np

# Test 1: Load URDF in MuJoCo
try:
    import mujoco
    model = mujoco.MjModel.from_xml_path("crx25ia.urdf")
    data = mujoco.MjData(model)
    print(f"MuJoCo loaded: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators, {model.nbody} bodies")
    
    # Print body names
    for i in range(model.nbody):
        name = model.body(i).name
        print(f"  body {i}: {name}")
    
    # Print joint names
    for i in range(model.njnt):
        name = model.joint(i).name
        print(f"  joint {i}: {name}")
    
    # FK at home position (all zeros)
    mujoco.mj_forward(model, data)
    tool0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool0")
    print(f"\nHome TCP: {data.xpos[tool0_id] * 1000} mm")
    
    # FK at the Fanuc test angles (need to apply offsets)
    # Fanuc angles: [6.34, -1.75, -21.95, -2.22, -83.66, 28.51] deg
    # URDF offsets: J2+31.1°, J3-16.7°
    q_fanuc = np.deg2rad([6.34, -1.75, -21.95, -2.22, -83.66, 28.51])
    q_urdf = q_fanuc.copy()
    q_urdf[1] += np.deg2rad(31.10)
    q_urdf[2] += np.deg2rad(-16.73)
    
    data.qpos[:6] = q_urdf
    mujoco.mj_forward(model, data)
    tcp = data.xpos[tool0_id] * 1000
    print(f"Test TCP:     ({tcp[0]:.2f}, {tcp[1]:.2f}, {tcp[2]:.2f}) mm")
    print(f"Expected TCP: (631.48, -108.82, 489.35) mm")
    print(f"Error: {np.linalg.norm(tcp - [631.48, -108.82, 489.35]):.2f} mm")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
