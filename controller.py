import numpy as np
from numpy.typing import ArrayLike

# ======================
# Tuning knobs
# ======================

# P controlled gains for tuning knobs
STEER_GAIN = 3.8         # P-gain for steering rate - how aggressively the car fixes steering errors
ACCEL_GAIN = 4.2         # P-gain for longitudinal acceleration - how strongly the car tries to match desired speed

# determines how far ahead the steering controller aims
# A small lookahead → tighter turns
# A large lookahead → smoother high-speed behavior
MIN_LOOKAHEAD = 3
MAX_LOOKAHEAD = 20
LOOKAHEAD_SCALING = 0.22

YAW_CORRECTION_TIME = 0.37     

MAX_VEL = 100.0
MIN_VEL = 5.0

CURV_LOOKAHEAD_DIST = 140       
CURV_STEP = 4                   
MAX_LAT_ACC = 9.0               
BRAKE_ACC = 7.2                 

def wrap(theta: float) -> float:
    """Normalize angle to [-pi, pi]. Used for heading error so the car doesn't wrap around incorrectly."""
    return np.arctan2(np.sin(theta), np.cos(theta))


def nearest_track_index(pos: np.ndarray, pts: np.ndarray) -> int:
    """Return the index of the centerline point closest to the car position."""
    car_xy = pos[:2]
    track_xy = pts[:, :2]

    diff = track_xy - car_xy
    squared_distances = np.sum(diff ** 2, axis=1)
    closest_index = int(np.argmin(squared_distances))

    return closest_index


def sample_radius(p1, p2, p3):
    """Estimate turn radius from three points."""
    v1 = p2 - p1
    v2 = p3 - p2
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-3 or n2 < 1e-3:
        return 1e5  # nearly straight

    cos_ang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    ang = np.arccos(cos_ang)

    if ang < 0.03:
        return 1e5

    return n1 / ang

# SPEED PLANNER (refactored)

def target_speed(state: np.ndarray, track, params: np.ndarray) -> float:
    """Computes the target speed based on approximate curvature ahead."""

    car_xy = state[:2] # extract the car's position from the state vector
    cl = track.centerline[:, :2] # get the centerline points of the track
    N = cl.shape[0]

    idx0 = nearest_track_index(car_xy, cl) # find the closest point on the centerline to the car

    safe_v = MAX_VEL

    # Look ahead up to ~140 points to detect curves
    for offset in range(0, CURV_LOOKAHEAD_DIST, CURV_STEP):
        a = cl[(idx0 + offset) % N]
        b = cl[(idx0 + offset + CURV_STEP) % N]
        c = cl[(idx0 + offset + 2 * CURV_STEP) % N]

        R = sample_radius(a, b, c)
        v_corner = np.sqrt(MAX_LAT_ACC * R)

        dist = np.linalg.norm(a - car_xy)
        v_now = np.sqrt(max(0.0, v_corner**2 + 2 * BRAKE_ACC * dist)) # Compute braking safe speed at current position

        safe_v = min(safe_v, v_now) # take the lowest safe speed over the entire lookahead.

    return float(np.clip(safe_v, MIN_VEL, MAX_VEL))

# STEERING TARGET (refactored)

def steering_target(state: np.ndarray, track, params: np.ndarray) -> float:
    """Computes the desired steering angle that turns the car toward a forward point."""

    car_xy = state[:2] 
    heading = state[4] 
    speed = max(state[3], 1.0) 

    cl = track.centerline[:, :2]
    wb = params[0]
    max_delta = params[4] 

    raw_LA = LOOKAHEAD_SCALING * speed
    LA = int(np.clip(raw_LA, MIN_LOOKAHEAD, MAX_LOOKAHEAD))

    idx0 = nearest_track_index(car_xy, cl)
    aim_pt = cl[(idx0 + LA) % cl.shape[0]]

    dx = aim_pt[0] - car_xy[0]
    dy = aim_pt[1] - car_xy[1]

    desired_heading = np.arctan2(dy, dx)
    heading_err = wrap(desired_heading - heading)

    desired_yaw_rate = heading_err / YAW_CORRECTION_TIME

    delta = np.arctan((wb * desired_yaw_rate) / speed)

    return float(np.clip(delta, -max_delta, max_delta))

# HIGH-LEVEL CONTROLLER

def controller(state: ArrayLike, params: ArrayLike, track) -> np.ndarray:
    """
    Outputs:
        [desired steering angle, desired speed]
    """
    s = np.asarray(state, float)
    p = np.asarray(params, float)

    desired_delta = steering_target(s, track, p)
    desired_vel = target_speed(s, track, p)

    return np.array([desired_delta, desired_vel], dtype=float)

# LOW-LEVEL CONTROLLER

def lower_controller(state: ArrayLike, ref: ArrayLike, params: ArrayLike) -> np.ndarray:
    """
    Converts:
        desired steering angle, desired speed
    into:
        steering_rate, acceleration
    using simple but tuned P controllers.
    """

    s = np.asarray(state, float)
    r = np.asarray(ref, float)

    cur_delta = s[2]
    cur_vel   = s[3]

    delta_ref = r[0]
    vel_ref   = r[1]

    # Steering rate cmd
    steer_err = wrap(delta_ref - cur_delta)
    steer_rate = STEER_GAIN * steer_err

    # Acceleration cmd
    vel_err = vel_ref - cur_vel
    accel_cmd = ACCEL_GAIN * vel_err

    return np.array([steer_rate, accel_cmd], dtype=float)