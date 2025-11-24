import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack


# ================================================================
# Helper utilities
# ================================================================

def _wrap_angle(theta: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return np.arctan2(np.sin(theta), np.cos(theta))


def _closest_centerline_index(position: ArrayLike, racetrack: RaceTrack) -> int:
    """Return the index of the closest point on the track centerline."""
    diffs = racetrack.centerline[:, :2] - position[:2]
    dists_sq = np.einsum("ij,ij->i", diffs, diffs)
    return int(np.argmin(dists_sq))


def _compute_curvature(points: ArrayLike, idx: int) -> float:
    """Compute local path curvature using Menger curvature."""
    n = len(points)
    idx_prev = (idx - 1) % n
    idx_next = (idx + 1) % n
    
    p1 = points[idx_prev, :2]
    p2 = points[idx, :2]
    p3 = points[idx_next, :2]
    
    area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) -
                     (p3[0] - p1[0]) * (p2[1] - p1[1]))
    
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p3 - p1)
    
    if a < 1e-6 or b < 1e-6 or c < 1e-6:
        return 0.0
    
    return 4.0 * area / (a * b * c)


# ================================================================
# PID Controller
# ================================================================

class PIDController:
    def __init__(self, kp, ki, kd, output_limits=None, windup_limit=10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.windup_limit = windup_limit
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False
    
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False
    
    def compute(self, error, dt=0.1):
        p = self.kp * error
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.windup_limit, self.windup_limit)
        i = self.ki * self.integral
        
        if not self.initialized:
            d = 0.0
            self.initialized = True
        else:
            d = self.kd * (error - self.prev_error) / dt
        
        self.prev_error = error
        
        out = p + i + d
        if self.output_limits is not None:
            out = np.clip(out, self.output_limits[0], self.output_limits[1])
        return out


_speed_controller = None
_steering_controller = None


# ================================================================
# Low-level controller
# ================================================================

def lower_controller(state, desired, parameters):
    """PID tracking of steering and speed."""
    global _speed_controller, _steering_controller
    
    state = np.asarray(state, float)
    desired = np.asarray(desired, float)
    parameters = np.asarray(parameters, float)

    delta = state[2]
    v = state[3]
    delta_ref, v_ref = desired
    
    delta_dot_min = parameters[7]
    a_min = parameters[8]
    delta_dot_max = parameters[9]
    a_max = parameters[10]

    if _speed_controller is None:
        _speed_controller = PIDController(
            kp=2.0, ki=0.4, kd=0.1,
            output_limits=(a_min, a_max),
            windup_limit=20
        )
    if _steering_controller is None:
        _steering_controller = PIDController(
            kp=3.3, ki=0.0, kd=0.3,
            output_limits=(delta_dot_min, delta_dot_max),
            windup_limit=5
        )

    # Steering
    delta_error = _wrap_angle(delta_ref - delta)
    delta_dot = _steering_controller.compute(delta_error, 0.1)

    # Speed
    v_error = v_ref - v
    v_dot = _speed_controller.compute(v_error, 0.1)

    delta_dot = np.clip(delta_dot, delta_dot_min, delta_dot_max)
    v_dot = np.clip(v_dot, a_min, a_max)
    return np.array([delta_dot, v_dot])


# ================================================================
# High-level controller
# ================================================================

def controller(state, parameters, racetrack: RaceTrack):
    state = np.asarray(state, float)
    parameters = np.asarray(parameters, float)

    x, y, delta, v, phi = state
    
    wheelbase = parameters[0]
    delta_min = parameters[1]
    v_min = parameters[2]
    delta_max = parameters[4]
    v_max = parameters[5]

    centerline = racetrack.centerline
    n = centerline.shape[0]
    position = state[:2]

    # ------------------------------------------------------------
    # 1. Closest point
    # ------------------------------------------------------------
    idx_closest = _closest_centerline_index(position, racetrack)

    # ------------------------------------------------------------
    # 2. Short-range curvature (immediate)
    # ------------------------------------------------------------
    max_curv = 0.0
    for i in range(8):
        idx = (idx_closest + 2 * i) % n
        max_curv = max(max_curv, _compute_curvature(centerline, idx))

    # ------------------------------------------------------------
    # 3. Long-range curvature (anticipatory braking)
    # ------------------------------------------------------------
    long_range_curv = 0.0
    for i in range(20):
        idx = (idx_closest + 3 * i) % n
        long_range_curv = max(long_range_curv, _compute_curvature(centerline, idx))

    effective_curv = max(max_curv, long_range_curv)

    # Curvature thresholds
    tight_thresh = 0.015
    medium_thresh = 0.008
    gentle_thresh = 0.004

    # ------------------------------------------------------------
    # 4. Adaptive lookahead (smaller in turns)
    # ------------------------------------------------------------
    if effective_curv > tight_thresh:
        LA = 5
    elif effective_curv > medium_thresh:
        LA = 8
    elif effective_curv > gentle_thresh:
        LA = 12
    else:
        base_LA = 14
        LA = int(base_LA * np.clip(v / v_max, 0.3, 1.0))
        LA = int(np.clip(LA, 10, 25))

    idx_LA = (idx_closest + LA) % n
    target = centerline[idx_LA, :2]

    # Transform to vehicle frame
    dx = target[0] - x
    dy = target[1] - y
    cp, sp = np.cos(phi), np.sin(phi)
    
    x_local = cp * dx + sp * dy
    y_local = -sp * dx + cp * dy
    Ld = np.hypot(x_local, y_local)

    # ------------------------------------------------------------
    # 5. Pure Pursuit steering
    # ------------------------------------------------------------
    if Ld < 0.5:
        delta_ref = delta
    else:
        alpha = np.arctan2(y_local, x_local)
        curvature_cmd = 2 * np.sin(alpha) / max(Ld, 1e-6)
        delta_ref = np.arctan(wheelbase * curvature_cmd)
        delta_ref = np.clip(delta_ref, delta_min, delta_max)

    # ------------------------------------------------------------
    # 6. Hard steering-based slowdown
    # ------------------------------------------------------------
    if abs(delta_ref) > 0.70 * delta_max:
        v_max_turn = 0.12 * v_max
    elif abs(delta_ref) > 0.85 * delta_max:
        v_max_turn = 0.08 * v_max
    else:
        v_max_turn = v_max

    # ------------------------------------------------------------
    # 7. Conservative speed mapping (VERY SLOW in turns)
    # ------------------------------------------------------------
    straight_speed = 0.60 * v_max
    gentle_speed   = 0.40 * v_max
    medium_speed   = 0.25 * v_max
    tight_speed    = 0.10 * v_max

    if effective_curv > tight_thresh:
        v_ref = tight_speed
    elif effective_curv > medium_thresh:
        v_ref = medium_speed
    elif effective_curv > gentle_thresh:
        v_ref = gentle_speed
    else:
        v_ref = straight_speed

    # Also enforce steering-limited max speed
    v_ref = min(v_ref, v_max_turn)

    # ------------------------------------------------------------
    # 8. Heading alignment penalty
    # ------------------------------------------------------------
    ahead = centerline[(idx_closest + 2) % n, :2]
    curr = centerline[idx_closest, :2]
    
    path_heading = np.arctan2(ahead[1] - curr[1], ahead[0] - curr[0])
    heading_err = abs(_wrap_angle(path_heading - phi))

    if heading_err > np.radians(35):
        v_ref *= 0.45
    elif heading_err > np.radians(20):
        v_ref *= 0.65
    elif heading_err > np.radians(10):
        v_ref *= 0.85

    # ------------------------------------------------------------
    # 9. Ensure bounds
    # ------------------------------------------------------------
    v_ref = np.clip(v_ref, max(0.1, v_min), v_max)

    return np.array([delta_ref, v_ref], float)
