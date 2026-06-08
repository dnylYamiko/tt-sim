import numpy as np


# ============================================================
# 1. TRACKING PERFORMANCE
# ============================================================

def compute_tracking_metrics(ee, target):
    ee = np.asarray(ee, dtype=float)
    target = np.asarray(target, dtype=float)

    error = np.linalg.norm(ee - target, axis=1)

    return {
        "error": error,
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(error)),
        "max_error": float(np.max(error)),
        "final_error": float(error[-1])
    }


# ============================================================
# 2. INTERCEPTION PERFORMANCE (BALL vs EE)
# ============================================================

def compute_interception_error(ee, ball):
    ee = np.asarray(ee, dtype=float)
    ball = np.asarray(ball, dtype=float)

    min_len = min(len(ee), len(ball))

    ee = ee[:min_len]
    ball = ball[:min_len]

    errors = np.linalg.norm(ee - ball, axis=1)

    hit_index = int(np.argmin(errors))

    return {
        "interception_error": float(errors[hit_index]),
        "hit_index": hit_index,
        "min_distance": float(np.min(errors))
    }


# ============================================================
# 3. CONTROL EFFORT
# ============================================================

def compute_control_effort(q_cmd):
    q_cmd = np.asarray(q_cmd, dtype=float)

    effort = np.sum(q_cmd ** 2)

    return {
        "control_effort": float(effort)
    }


# ============================================================
# 4. CONTROL SMOOTHNESS (Δu)
# ============================================================

def compute_smoothness(q_cmd):
    q_cmd = np.asarray(q_cmd, dtype=float)

    if len(q_cmd) < 2:
        return {"smoothness": 0.0}

    dq = np.diff(q_cmd, axis=0)

    smoothness = np.sum(np.linalg.norm(dq, axis=1) ** 2)

    return {
        "smoothness": float(smoothness)
    }


# ============================================================
# 5. COST STATISTICS (MPC COST)
# ============================================================

def compute_cost_stats(cost):
    cost = np.asarray(cost, dtype=float)

    return {
        "mean_cost": float(np.mean(cost)),
        "min_cost": float(np.min(cost)),
        "max_cost": float(np.max(cost))
    }


# ============================================================
# 6. SUCCESS RATE
# ============================================================

def compute_success_rate(error, threshold=0.05):
    error = np.asarray(error, dtype=float)
    threshold = float(threshold)

    success = error <= threshold

    return {
        "success_rate_pct": float(100.0 * np.mean(success)),
        "threshold": threshold
    }


# ============================================================
# 7. MAIN EVALUATION FUNCTION (USED BY YOUR SIMULATION)
# ============================================================

def evaluate_run(logs, success_threshold=0.05):
    """
    Full evaluation of a single MPC/SINDy run
    """

    ee = np.asarray(logs["ee"], dtype=float)
    target = np.asarray(logs["target"], dtype=float)
    q_cmd = np.asarray(logs["q_cmd"], dtype=float)
    cost = np.asarray(logs["cost"], dtype=float)
    ball = np.asarray(logs["ball"], dtype=float)

    # ---- tracking ----
    tracking = compute_tracking_metrics(ee, target)

    # ---- interception ----
    interception = compute_interception_error(ee, ball)

    # ---- control ----
    effort = compute_control_effort(q_cmd)
    smooth = compute_smoothness(q_cmd)

    # ---- cost ----
    cost_stats = compute_cost_stats(cost)

    # ---- success ----
    success = compute_success_rate(
        tracking["error"],
        success_threshold
    )

    return {
        "tracking": tracking,
        "interception": interception,
        "control": effort,
        "smoothness": smooth,
        "cost": cost_stats,
        "success": success
    }


# ============================================================
# 8. COMPARISON FUNCTION (BASELINE vs SINDY)
# ============================================================

def compare_runs(baseline_logs, sindy_logs, success_threshold=0.05):

    base = evaluate_run(baseline_logs, success_threshold)
    sindy = evaluate_run(sindy_logs, success_threshold)

    def flatten(d):
        return {
            "rmse": d["tracking"]["rmse"],
            "mae": d["tracking"]["mae"],
            "max_error": d["tracking"]["max_error"],
            "final_error": d["tracking"]["final_error"],
            "success_rate": d["success"]["success_rate_pct"],
            "control_effort": d["control"]["control_effort"],
            "smoothness": d["smoothness"]["smoothness"],
            "mean_cost": d["cost"]["mean_cost"]
        }

    return {
        "baseline": flatten(base),
        "sindy": flatten(sindy)
    }


# ============================================================
# 9. PRINT FRIENDLY TABLE
# ============================================================

def print_comparison_table(baseline, sindy):

    print("\n========== MPC vs SINDy-MPC COMPARISON ==========\n")

    keys = baseline.keys()

    print(f"{'Metric':25s} {'Baseline':15s} {'SINDy':15s}")

    for k in keys:
        print(
            f"{k:25s} {baseline[k]:15.6f} {sindy[k]:15.6f}"
        )