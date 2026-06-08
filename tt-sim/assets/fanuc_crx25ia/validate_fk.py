#!/usr/bin/env python3
"""Refine FK with constrained nice-number search.

From optimizer: d1=404.66, off2=31.1°, off3=-16.73°, ey=-183.28
These should correspond to physical dimensions. Let me try:
- d1 around 400mm 
- Combined off2+off3 might relate to standard DH offsets
- ey might be a real structural offset

Also need to verify with orientation (WPR = -170.56, 12.67, 35.36).
"""
import numpy as np
from scipy.optimize import minimize

J_RAD = np.deg2rad([6.34, -1.75, -21.95, -2.22, -83.66, 28.51])
TCP_MM = np.array([631.48, -108.82, 489.35])
TCP_WPR = np.deg2rad([-170.56, 12.67, 35.36])  # W, P, R

def rot(axis, t):
    c,s = np.cos(t), np.sin(t)
    if axis=='z': return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])
    if axis=='y': return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]])
    if axis=='x': return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]])

def tr(x,y,z):
    T=np.eye(4); T[:3,3]=[x,y,z]; return T

# Best-fit model
def fk(q_f, d1=404.66, off2=np.deg2rad(31.10), off3=np.deg2rad(-16.73), ey=-183.28):
    q = q_f.copy()
    q[1] += off2; q[2] += off3
    T = tr(0,0,d1) @ rot('z',q[0])
    T = T @ rot('y',q[1])
    T = T @ tr(0,0,950) @ tr(0,ey,0) @ rot('y',-q[2])
    T = T @ rot('x',-q[3])
    T = T @ tr(730,0,0) @ rot('y',-q[4])
    T = T @ tr(140,0,0) @ rot('x',-q[5])
    return T

T = fk(J_RAD)
R = T[:3,:3]
print("FK position:", T[:3,3])
print("FK rotation matrix:")
print(R)

# Fanuc WPR -> rotation matrix (ZYX Euler: W=Rz, P=Ry, R=Rx)
w, p, r = TCP_WPR
R_expected = rot('z', w)[:3,:3] @ rot('y', p)[:3,:3] @ rot('x', r)[:3,:3]
print("\nExpected rotation (from WPR):")
print(R_expected)

# Without tool0 transform, our flange frame orientation
print("\nOrientation error (Frobenius):", np.linalg.norm(R - R_expected))

# The tool frame transform might account for the difference
# CRX-30iA has tool0: rpy=(pi, -pi/2, 0)
# Let's check if adding that transform fixes the orientation
T_tool = rot('z', np.pi) @ rot('y', -np.pi/2)
T_with_tool = T @ T_tool
R_with_tool = T_with_tool[:3,:3]
print("\nWith CRX-30iA tool0 transform:")
print(R_with_tool)
print("Orientation error:", np.linalg.norm(R_with_tool - R_expected))

# Check: what's the actual off2/off3 in nice degrees?
# off2=31.1° is close to atan(ey/a2) = atan(183.28/950) = 10.9°... nope
# Maybe the convention is different. Let me check if we can use 
# a different parameterization that gives nicer numbers.

# Standard Fanuc DH: typically has a2=upper_arm, a3=elbow_offset, d4=forearm
# Let me try DH parameters
print("\n\n=== Trying standard DH parameterization ===")
# Modified DH: alpha_{i-1}, a_{i-1}, d_i, theta_offset
# For 6R robot like Fanuc:
# Joint 1: alpha0=0, a0=0, d1=?, theta1=q1
# Joint 2: alpha1=-90, a1=0, d2=0, theta2=q2+off2
# Joint 3: alpha2=0, a2=950, d3=0, theta3=q3+off3
# Joint 4: alpha3=-90, a3=a3_offset, d4=d4, theta4=q4
# Joint 5: alpha4=90, a4=0, d5=0, theta5=q5
# Joint 6: alpha5=-90, a5=0, d6=0, theta6=q6

def dh_transform(alpha, a, d, theta):
    ca,sa = np.cos(alpha), np.sin(alpha)
    ct,st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0, a],
        [st*ca, ct*ca, -sa, -d*sa],
        [st*sa, ct*sa, ca, d*ca],
        [0, 0, 0, 1]
    ])

def fk_dh(q, params):
    d1, off2, off3, a3, d4 = params
    T = dh_transform(0, 0, d1, q[0])
    T = T @ dh_transform(-np.pi/2, 0, 0, q[1]+off2)
    T = T @ dh_transform(0, 950, 0, q[2]+off3)
    T = T @ dh_transform(-np.pi/2, a3, d4, q[3])
    T = T @ dh_transform(np.pi/2, 0, 0, q[4])
    T = T @ dh_transform(-np.pi/2, 0, 0, q[5])
    return T

def err_dh(params):
    T = fk_dh(J_RAD, params)
    return np.linalg.norm(T[:3,3] - TCP_MM)

# Search DH params
best = (1e9, None)
for d1 in np.arange(300, 450, 10):
    for off2 in np.deg2rad(np.arange(-180, 181, 15)):
        for off3 in np.deg2rad(np.arange(-180, 181, 15)):
            for a3 in np.arange(-200, 201, 50):
                for d4 in np.arange(500, 900, 50):
                    p = [d1, off2, off3, a3, d4]
                    e = err_dh(p)
                    if e < best[0]:
                        best = (e, p[:])

print(f"Grid: d1={best[1][0]:.0f} off2={np.rad2deg(best[1][1]):.0f}° off3={np.rad2deg(best[1][2]):.0f}° a3={best[1][3]:.0f} d4={best[1][4]:.0f} err={best[0]:.2f}mm")

if best[0] < 100:
    res = minimize(err_dh, best[1], method='Nelder-Mead', options={'xatol':0.01,'fatol':0.01,'maxiter':20000})
    p = res.x
    print(f"Opt:  d1={p[0]:.2f} off2={np.rad2deg(p[1]):.2f}° off3={np.rad2deg(p[2]):.2f}° a3={p[3]:.2f} d4={p[4]:.2f} err={res.fun:.4f}mm")
    T = fk_dh(J_RAD, p)
    print(f"Computed: ({T[0,3]:.2f}, {T[1,3]:.2f}, {T[2,3]:.2f})")
