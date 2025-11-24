import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack


# ---------- Helper utilities ----------

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
    Compute local path curvature at index using finite differences.
    Uses the Menger curvature formula for three points.
    """
    n = len(points)
    idx_prev = (idx - 1) % n
    idx_next = (idx + 1) % n
    
    p1 = points[idx_prev, :2]
    p2 = points[idx, :2]
    p3 = points[idx_next, :2]
    
    # Menger curvature: k = 4*Area / (a*b*c)
    # Area of triangle formed by three points
    area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                      (p3[0] - p1[0]) * (p2[1] - p1[1]))
    
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p3 - p1)
    
    if a < 1e-6 or b < 1e-6 or c < 1e-6:
        return 0.0
    
    curvature = 4.0 * area / (a * b * c)
    return curvature


# ---------- PID Controllers ----------

class PIDController:
    """Generic PID controller with anti-windup."""
    
    def __init__(self, kp: float, ki: float, kd: float, 
                 output_limits: tuple = None, windup_limit: float = 10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.windup_limit = windup_limit
        
        # State variables
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False
    
    def compute(self, error: float, dt: float = 0.1) -> float:
        """
        Compute PID control output.
        
        Args:
            error: setpoint - measurement
            dt: time step
            
        Returns:
            control output
        """
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.windup_limit, self.windup_limit)
        i_term = self.ki * self.integral
        
        # Derivative term (with initialization check)
        if not self.initialized:
            d_term = 0.0
            self.initialized = True
        else:
            derivative = (error - self.prev_error) / dt
            d_term = self.kd * derivative
        
        self.prev_error = error
        
        # Compute output
        output = p_term + i_term + d_term
        
        # Apply output limits
        if self.output_limits is not None:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        return output


# Global controller instances (maintained across calls)
_speed_controller = None
_steering_controller = None


# ---------- Low-level controller ----------
# Maps desired [steering_angle, speed] -> inputs [steering_rate, acceleration]

def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    """
    Track desired steering angle and speed with PID controllers.

    state:      [sx, sy, delta, v, phi]
    desired:    [delta_ref, v_ref]
    parameters: packed vehicle parameters (see racecar.py)

    returns:    [delta_dot, v_dot]
    """
    global _speed_controller, _steering_controller
    
    state = np.asarray(state, dtype=float)
    desired = np.asarray(desired, dtype=float)
    parameters = np.asarray(parameters, dtype=float)

    assert state.shape == (5,)
    assert desired.shape == (2,)
    assert parameters.shape == (11,)

    # Current states
    delta = state[2]
    v = state[3]

    delta_ref, v_ref = desired
    
    # Extract limits from parameters
    delta_dot_min = parameters[7]
    a_min = parameters[8]
    delta_dot_max = parameters[9]
    a_max = parameters[10]
    
    # Initialize controllers on first call
    if _speed_controller is None:
        # Longitudinal PI controller
        # Tuned for smooth acceleration with no overshoot
        _speed_controller = PIDController(
            kp=2.5,    # Proportional gain
            ki=0.5,    # Integral gain for steady-state error elimination
            kd=0.1,    # Small derivative for damping
            output_limits=(a_min, a_max),
            windup_limit=20.0
        )
    
    if _steering_controller is None:
        # Lateral PD controller
        # Higher gain for responsive steering, derivative for damping
        _steering_controller = PIDController(
            kp=3.5,    # High proportional gain for quick response
            ki=0.0,    # No integral (avoid windup in steering)
            kd=0.3,    # Derivative for damping oscillations
            output_limits=(delta_dot_min, delta_dot_max),
            windup_limit=5.0
        )

    # Compute steering rate (wrap angle error properly)
    delta_error = _wrap_angle(delta_ref - delta)
    delta_dot = _steering_controller.compute(delta_error, dt=0.1)

    # Compute longitudinal acceleration
    v_error = v_ref - v
    v_dot = _speed_controller.compute(v_error, dt=0.1)

    # Safety clips (redundant but ensures limits)
    delta_dot = np.clip(delta_dot, delta_dot_min, delta_dot_max)
    v_dot = np.clip(v_dot, a_min, a_max)

    u = np.array([delta_dot, v_dot], dtype=float)
    return u


# ---------- High-level controller ----------
# Computes desired [steering_angle, speed] from track geometry

def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    """
    High-level path-tracking controller using Pure Pursuit + adaptive speed.

    state:      [sx, sy, delta, v, phi]
    parameters: packed vehicle parameters
    racetrack:  RaceTrack object with .centerline
    
    returns:    [delta_ref, v_ref]
    """
    state = np.asarray(state, dtype=float)
    parameters = np.asarray(parameters, dtype=float)

    assert state.shape == (5,)
    assert parameters.shape == (11,)

    # Unpack state
    x, y, delta, v, phi = state
    
    # Unpack parameters
    wheelbase = parameters[0]
    v_min = parameters[2]
    v_max = parameters[5]
    delta_min = parameters[1]
    delta_max = parameters[4]

    position = state[0:2]
    n_points = racetrack.centerline.shape[0]

    # ========== Step 1: Find closest point on centerline ==========
    idx_closest = _closest_centerline_index(position, racetrack)

    # ========== Step 2: Adaptive lookahead distance ==========
    # Use velocity-dependent lookahead for better performance
    # Faster -> look further ahead, Slower -> look closer
    base_lookahead = 15.0  # meters
    speed_factor = np.clip(v / v_max, 0.3, 1.0)
    lookahead_distance = base_lookahead * speed_factor
    
    # Convert distance to number of points (approximate)
    avg_point_spacing = 1.0  # Assume ~1m between points
    lookahead_points = int(lookahead_distance / avg_point_spacing)
    lookahead_points = np.clip(lookahead_points, 10, 40)
    
    idx_lookahead = (idx_closest + lookahead_points) % n_points
    target_point = racetrack.centerline[idx_lookahead, :2]

    # ========== Step 3: Transform to vehicle frame ==========
    dx = target_point[0] - x
    dy = target_point[1] - y

    # Rotate into car frame (forward = x, left = y)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    x_local = cos_phi * dx + sin_phi * dy
    y_local = -sin_phi * dx + cos_phi * dy

    lookahead_dist = np.hypot(x_local, y_local)

    # ========== Step 4: Pure Pursuit steering angle ==========
    if lookahead_dist < 0.5:
        # Too close to target, maintain current steering
        delta_ref = delta
    else:
        # Pure pursuit formula: delta = atan(2 * L * sin(alpha) / Ld)
        alpha = np.arctan2(y_local, x_local)
        
        # Curvature-based steering (pure pursuit)
        curvature = 2.0 * np.sin(alpha) / lookahead_dist
        
        # Convert curvature to steering angle using bicycle model
        delta_ref = np.arctan(wheelbase * curvature)
        
        # Apply steering limits
        delta_ref = np.clip(delta_ref, delta_min, delta_max)

    # ========== Step 5 & 6: Adaptive speed based on curvature ==========
    # Compute local curvature at multiple points ahead for preview
    curvature_lookahead = 0.0
    num_preview_points = 5
    
    for i in range(num_preview_points):
        preview_idx = (idx_closest + i * 3) % n_points
        local_curv = _compute_curvature(racetrack.centerline, preview_idx)
        curvature_lookahead = max(curvature_lookahead, local_curv)
    
    # Also consider current steering angle as indicator of turn severity
    abs_delta = abs(delta)
    
    # Speed profile based on curvature and steering angle
    # High curvature or large steering -> slow down
    # Low curvature and small steering -> speed up
    
    # Base speeds
    max_corner_speed = 0.30 * v_max  # 30% of max in tight corners
    max_straight_speed = 0.85 * v_max  # 85% of max on straights
    
    # Curvature threshold for determining corner vs straight
    curvature_threshold = 0.02  # rad/m
    
    # Compute speed based on curvature
    if curvature_lookahead < curvature_threshold * 0.3:
        # Essentially straight
        curvature_speed = max_straight_speed
    elif curvature_lookahead < curvature_threshold:
        # Gentle curve
        ratio = curvature_lookahead / curvature_threshold
        curvature_speed = max_straight_speed * (1.0 - 0.3 * ratio)
    else:
        # Sharp corner - slow down based on severity
        ratio = min(curvature_lookahead / (curvature_threshold * 3.0), 1.0)
        curvature_speed = max_corner_speed + (max_straight_speed - max_corner_speed) * (1.0 - ratio)
    
    # Also reduce speed based on current steering angle
    # Large |delta| means we're turning -> reduce speed
    steering_factor = 1.0 - 0.4 * (abs_delta / delta_max)
    steering_factor = max(steering_factor, 0.3)
    
    # Combine both factors
    v_ref = curvature_speed * steering_factor
    
    # Additionally: reduce speed if heading error is large
    # This helps when recovering from deviations
    if idx_closest < n_points - 1:
        p_ahead = racetrack.centerline[(idx_closest + 2) % n_points, :2]
        p_current = racetrack.centerline[idx_closest, :2]
    else:
        p_ahead = racetrack.centerline[1, :2]
        p_current = racetrack.centerline[0, :2]
    
    path_heading = np.arctan2(p_ahead[1] - p_current[1], 
                               p_ahead[0] - p_current[0])
    heading_error = abs(_wrap_angle(path_heading - phi))
    
    # Penalize speed if heading error is large
    if heading_error > np.pi / 6:  # > 30 degrees
        heading_penalty = 0.6
    elif heading_error > np.pi / 12:  # > 15 degrees
        heading_penalty = 0.8
    else:
        heading_penalty = 1.0
    
    v_ref = v_ref * heading_penalty
    
    # Ensure within absolute bounds
    v_ref = np.clip(v_ref, max(0.1, v_min), v_max)

    # ========== Step 7: Return desired state ==========
    desired = np.array([delta_ref, v_ref], dtype=float)
    return desired