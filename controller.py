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
    """
    Compute local path curvature at index using Menger curvature.
    """
    n = len(points)
    idx_prev = (idx - 1) % n
    idx_next = (idx + 1) % n
    
    p1 = points[idx_prev, :2]
    p2 = points[idx, :2]
    p3 = points[idx_next, :2]
    
    # Triangle area
    area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) -
                     (p3[0] - p1[0]) * (p2[1] - p1[1]))
    
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p3 - p1)
    
    if a < 1e-6 or b < 1e-6 or c < 1e-6:
        return 0.0
    
    curvature = 4.0 * area / (a * b * c)
    return curvature


# ================================================================
#       PID Controller class
# ================================================================

class PIDController:
    """Generic PID controller with anti-windup."""
    
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
        p_term = self.kp * error
        
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.windup_limit, self.windup_limit)
        i_term = self.ki * self.integral
        
        if not self.initialized:
            d_term = 0.0
            self.initialized = True
        else:
            d_term = self.kd * ((error - self.prev_error) / dt)
        
        self.prev_error = error
        
        output = p_term + i_term + d_term
        
        if self.output_limits is not None:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        return output


_speed_controller = None
_steering_controller = None


# ================================================================
# Low-level controller
# ================================================================

def lower_controller(state, desired, parameters):
    """
    Low-level PID control of steering rate and acceleration.
    """
    global _speed_controller, _steering_controller
    
    state = np.asarray(state, float)
    desired = np.asarray(desired, float)
    parameters = np.asarray(parameters, float)

    delta = state[2]
    v = state[3]

    delta_ref, v_ref = desired
    
    # Extract limits
    delta_dot_min = parameters[7]
    a_min = parameters[8]
    delta_dot_max = parameters[9]
    a_max = parameters[10]

    # Initialize controllers if needed
    if _speed_controller is None:
        _speed_controller = PIDController(
            kp=2.0,
            ki=0.4,
            kd=0.1,
            output_limits=(a_min, a_max),
            windup_limit=20.0
        )
    if _steering_controller is None:
        _steering_controller = PIDController(
            kp=3.3,
            ki=0.0,
            kd=0.3,
            output_limits=(delta_dot_min, delta_dot_max),
            windup_limit=5.0
        )

    # Steering rate command
    delta_error = _wrap_angle(delta_ref - delta)
    delta_dot = _steering_controller.compute(delta_error, dt=0.1)

    # Acceleration command
    v_error = v_ref - v
    v_dot = _speed_controller.compute(v_error, dt=0.1)

    # Safety clip
    delta_dot = np.clip(delta_dot, delta_dot_min, delta_dot_max)
    v_dot = np.clip(v_dot, a_min, a_max)

    return np.array([delta_dot, v_dot])


# ================================================================
# High-level controller
# ================================================================

def controller(state, parameters, racetrack: RaceTrack):
    """
    High-level control using Pure Pursuit + extremely conservative
    curvature-based speed for perfect staying on track.
    """
    state = np.asarray(state, float)
    parameters = np.asarray(parameters, float)

    x, y, delta, v, phi = state
    
    wheelbase = parameters[0]
    delta_min = parameters[1]
    v_min = parameters[2]
    delta_max = parameters[4]
    v_max = parameters[5]

    centerline = racetrack.centerline
    n_points = centerline.shape[0]
    position = state[:2]

    # ------------------------------------------------------------
    # 1) Closest point on track
    # ------------------------------------------------------------
    idx_closest = _closest_centerline_index(position, racetrack)

    # ------------------------------------------------------------
    # 2) Dynamic lookahead
    # ------------------------------------------------------------
    base_LA = 14.0
    LA = int(base_LA * np.clip(v / v_max, 0.3, 1.0))
    LA = np.clip(LA, 10, 40)

    idx_LA = (idx_closest + LA) % n_points
    target = centerline[idx_LA, :2]

    # Transform to car frame
    dx = target[0] - x
    dy = target[1] - y
    cp = np.cos(phi)
    sp = np.sin(phi)
    x_local = cp * dx + sp * dy
    y_local = -sp * dx + cp * dy
    Ld = np.hypot(x_local, y_local)

    # ------------------------------------------------------------
    # 3) Pure Pursuit steering
    # ------------------------------------------------------------
    if Ld < 0.5:
        delta_ref = delta
    else:
        alpha = np.arctan2(y_local, x_local)
        curvature = 2 * np.sin(alpha) / max(Ld, 1e-6)
        delta_ref = np.arctan(wheelbase * curvature)
        delta_ref = np.clip(delta_ref, delta_min, delta_max)

    # ------------------------------------------------------------
    # 4) Track curvature for speed control
    # ------------------------------------------------------------
    max_curv = 0.0
    for i in range(8):  # more preview → anticipates turns early
        idx = (idx_closest + 2 * i) % n_points
        max_curv = max(max_curv, _compute_curvature(centerline, idx))

    # Extremely conservative thresholds for perfect tracking
    tight_thresh = 0.015
    medium_thresh = 0.008
    gentle_thresh = 0.004

    # Speed choices
    straight_speed = 0.75 * v_max
    gentle_speed   = 0.60 * v_max
    medium_speed   = 0.45 * v_max
    tight_speed    = 0.22 * v_max
    emergency_speed = 0.15 * v_max

    # Determine speed based on curvature
    if max_curv > tight_thresh:
        v_ref = tight_speed
    elif max_curv > medium_thresh:
        v_ref = medium_speed
    elif max_curv > gentle_thresh:
        v_ref = gentle_speed
    else:
        v_ref = straight_speed

    # ------------------------------------------------------------
    # 5) Additional speed reduction from steering + heading error
    # ------------------------------------------------------------
    steering_ratio = abs(delta) / delta_max
    if steering_ratio > 0.7:
        v_ref *= 0.55
    elif steering_ratio > 0.5:
        v_ref *= 0.70
    elif steering_ratio > 0.3:
        v_ref *= 0.85

    # Heading alignment penalty
    if idx_closest < n_points - 1:
        ahead = centerline[(idx_closest + 2) % n_points, :2]
    else:
        ahead = centerline[1, :2]
    current = centerline[idx_closest, :2]

    path_heading = np.arctan2(ahead[1] - current[1], ahead[0] - current[0])
    heading_error = abs(_wrap_angle(path_heading - phi))

    if heading_error > np.radians(35):
        v_ref *= 0.45
    elif heading_error > np.radians(20):
        v_ref *= 0.65
    elif heading_error > np.radians(10):
        v_ref *= 0.85

    # ------------------------------------------------------------
    # 6) Clip speed and return
    # ------------------------------------------------------------
    v_ref = np.clip(v_ref, max(0.1, v_min), v_max)

    return np.array([delta_ref, v_ref], float)
