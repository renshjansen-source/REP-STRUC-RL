''' This file contains the utility functions '''

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import math
from scipy.interpolate import interp1d

from internal_variables import IV
from environment.envs.BikeBuilder_Classes import Plane, BikeFrame, PointDict, Pairs, Vector2D, Point2D, ShapeGrammar

# =============================================================================
# FUNCTIONS - VECTOR AND POINT MANIPULATION
# =============================================================================

def cross(vector_a: Vector2D, vector_b: Vector2D) -> float:
    return vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]

def dot(vector_a: Vector2D, vector_b: Vector2D) -> float:
    return vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]
# CCP = CurveClosestPoint
def CCP(centroid: Point2D, sampled_curve: np.ndarray) -> tuple[int, np.float32]:
    distances   = np.linalg.norm(sampled_curve - centroid, axis = 1)
    nearest_idx = int(np.argmin(distances))
    return nearest_idx, distances[nearest_idx]

# =============================================================================
# FUNCTIONS - DATA HANDLING
# =============================================================================

def resample_curve(curve: np.ndarray, samples: int) -> np.ndarray:
    # Cumulative distance along the curve
    deltas = np.diff(curve, axis=0)                             # next point - current point
    distances = np.cumsum(np.linalg.norm(deltas, axis=1))       
    distances = np.insert(distances, 0, 0)

    # Normalizes the points to 't' parameter between 0 < 1
    distances /= distances[-1]

    # Interpolate x and z independently
    interpolator = interp1d(distances, curve, axis=0)

    # Sample evenly spaced points
    sampled_points = interpolator(np.linspace(0, 1, samples)).astype(np.float32)

    return sampled_points

def normalized_cross_sections(outer_diameter, thickness):
    inner_diameter = outer_diameter - 2.0 * thickness
    cross_section  = (math.pi / 4.0) * (outer_diameter**2 - inner_diameter**2)
    
    return cross_section

# =============================================================================
# FUNCTIONS - OPERATIONAL LOGICS
# =============================================================================

def distance_to(point_a, point_b) -> float:
    return float(np.linalg.norm(point_a - point_b))

def offset_plane(plane: Plane) -> Plane:
    normalized_Y = plane.Y_vector / np.linalg.norm(plane.Y_vector)
    new_origin = plane.Origin + normalized_Y * IV.connection_offset
    new_plane  = Plane(new_origin, plane.X_vector, plane.mirror)
    return new_plane

def rotation_angle(vector_a: Vector2D, vector_b: Vector2D) -> float:
    angle = np.arctan2(
            cross(vector_a, vector_b),
            dot(vector_a, vector_b)
    )    
    return angle

def rotation_matrix(angle: float):
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a],
                [sin_a,  cos_a]])
    return R

def reflection_matrix(vector: Vector2D) -> np.ndarray:
    # Reflection about the axis perpendicular to `vector` (vector must be unit length)
    nx, ny = -vector[1], vector[0]
    return np.array([
        [nx**2 - ny**2, 2*nx*ny],
        [2*nx*ny,       ny**2 - nx**2],
    ], dtype=np.float32)

def plane_transform(candidate_plane: Plane, target_plane: Plane) -> np.ndarray:
    candidate = np.column_stack([candidate_plane.X_vector, candidate_plane.Y_vector])
    target    = np.column_stack([target_plane.X_vector,   target_plane.Y_vector])

    candidate = candidate / np.linalg.norm(candidate, axis=0, keepdims=True)
    target    = target    / np.linalg.norm(target,    axis=0, keepdims=True)

    T = target @ candidate.T   # orthonormal basis → transpose is the inverse

    if candidate_plane.mirror:
        T = -T
    return T

def initial_targets(frame: BikeFrame, guide_curve):
    # Curve Tangent
    curve_start   = guide_curve[0]
    curve_tangent = guide_curve[0] - guide_curve[1]
    curve_normal  = np.array([-curve_tangent[1], curve_tangent[0]], dtype=np.float32)

    # Frame midline
    HT_idx = PointDict(1)
    CS_SS  = frame.points[4]
    HT_MID = frame.mid_pt(HT_idx)
    frame_midline = CS_SS - HT_MID

    # Angle between curve tangent and frame midline
    angle = rotation_angle(frame_midline, curve_tangent)
    R     = rotation_matrix(angle)

    # Moving the frame (temporarily)
    moved_midpoints = np.array([
        R @ (p - CS_SS) + curve_start
        for p in frame.mid_points
    ], dtype=np.float32)

    moved_vectors = np.array([
        R @ v
        for v in frame.mid_vectors
    ], dtype=np.float32)

    # Constructing option 1
    plane_1 = Plane(curve_start, curve_normal)

    # Construction options 2 and 3
    R_2     = rotation_matrix(IV.start_rotation)
    R_3     = rotation_matrix(-IV.start_rotation)
    vec_2   = R_2 @ curve_normal
    vec_3   = R_3 @ curve_normal
    plane_2 = Plane(curve_start, vec_2)
    plane_3 = Plane(curve_start, vec_3)

    # Constructing options 4 and 5
    plane_4 = Plane(moved_midpoints[3], -moved_vectors[3])
    plane_5 = Plane(moved_midpoints[4], -moved_vectors[4])

    initial_planes = [plane_1, plane_2, plane_3, plane_4, plane_5]

    return initial_planes

def place_first(
        frame         : BikeFrame, 
        initial_planes: list[Plane], 
        candidate     : PointDict, 
        target        : PointDict, 
        mirror        : bool
        ):
    
    target_plane    = initial_planes[target.value]
    candidate_plane = frame.get_candidate_plane(candidate, mirror)

    T = plane_transform(candidate_plane, target_plane)

    moved_points = np.array([
        T @ (p - candidate_plane.Origin) + target_plane.Origin
        for p in frame.points
    ], dtype=np.float32)

    moved_frame = BikeFrame(moved_points, recenter = False)

    return moved_frame

def place(
        frame          : BikeFrame,
        previous_frame : BikeFrame,
        candidate      : PointDict,
        target         : PointDict,
        mirror         : bool,
        mirror_flag    : bool,
        ):

    target_plane    = previous_frame.get_target_plane(target, mirror_flag)
    target_plane    = offset_plane(target_plane)
    candidate_plane = frame.get_candidate_plane(candidate, mirror)

    T = plane_transform(candidate_plane, target_plane)

    moved_points = np.array([
        T @ (p - candidate_plane.Origin) + target_plane.Origin
        for p in frame.points
    ], dtype=np.float32)

    moved_frame = BikeFrame(moved_points, recenter = False)

    return moved_frame
# =============================================================================
# FUNCTIONS - OBSERVATIONS
# =============================================================================

def encode_angles(angles: np.ndarray) -> np.ndarray:
    pairs = np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32)
    return np.round(pairs, decimals=IV.angle_rounding).astype(np.float32)

def build_observation_points(frame: BikeFrame, obs_type: str, norm_range) -> np.ndarray:
    if obs_type == "combined":
        return (frame.observation_points / norm_range).astype(np.float32)
    if obs_type == "edge":
        return (frame.points / norm_range).astype(np.float32)
    if obs_type == "mid":
        return (frame.mid_points / norm_range).astype(np.float32)
    if obs_type == "angle":
        normalized_points = frame.points / norm_range
        angle_encoded     = encode_angles(frame.turning_angles)
        n = len(frame.points)
        return np.stack([normalized_points, angle_encoded], axis=1).reshape(n * 2, 2).astype(np.float32)
    raise ValueError(f"Unknown obs_type {obs_type!r} - expected one of 'combined', 'edge', 'mid', 'angle'")

def build_current_frame_observation(placed_frames, obs_type, norm_range, points_per_frame, buffer_size=None):
    if buffer_size is None:
        if not placed_frames:
            return np.zeros((points_per_frame, 2), dtype=np.float32)
        return build_observation_points(placed_frames[-1], obs_type, norm_range)

    recent_frames = list(reversed(placed_frames[-buffer_size:]))  # most-recent-first
    stacked = np.zeros((buffer_size, points_per_frame, 2), dtype=np.float32)
    for i, frame in enumerate(recent_frames):
        stacked[i] = build_observation_points(frame, obs_type, norm_range)
    return stacked

# =============================================================================
# FUNCTIONS - PENALTIES
# =============================================================================

def segments_intersect(p1, p2, p3, p4) -> bool:
    d1 = p2 - p1
    d2 = p4 - p3
    cross_product = cross(d1, d2)

    if abs(cross_product) < IV.intersect_tol:
        return False # Guarding against 0 division

    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross_product
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross_product

    return 0.0 < t < 1.0 and 0.0 < u < 1.0

def frames_intersect(frame: BikeFrame, buffer_frames: list[BikeFrame]) -> bool:
    for other_frame in buffer_frames:
        for a1, a2 in Pairs.values():
            for b1, b2 in Pairs.values():
                if segments_intersect(
                    frame.points[a1], frame.points[a2],
                    other_frame.points[b1], other_frame.points[b2]
                ):
                    return True
    return False

# =============================================================================
# FUNCTIONS - REWARDS
# =============================================================================

def distance_reward(nearest_distance: np.float32, distance_weight: float) -> np.float32:
    return distance_weight * (1.0 - (nearest_distance / IV.distance_threshold))

def progression_reward(
        nearest_idx    : int,
        max_t          : np.float32,
        guide_curve    : np.ndarray,
        progress_weight: float,
) -> tuple[np.float32, np.float32]:

    t_current = np.float32(nearest_idx / IV.curve_samples)

    if t_current > max_t:
        progress_scale = (1.0 + t_current) * IV.progress_multiplier
        previous_curve_point_idx = int(max_t * IV.curve_samples)
        previous_curve_point     = guide_curve[previous_curve_point_idx]
        current_curve_point      = guide_curve[nearest_idx]

        progress = np.linalg.norm(current_curve_point - previous_curve_point)
        reward   = progress_weight * (progress / IV.progress_threshold) * progress_scale
        max_t    = t_current
    else: 
        reward   = IV.no_progress

    return np.float32(reward), np.float32(max_t)

def step_reward(
        frame: BikeFrame, 
        guide_curve: np.ndarray, 
        max_t: np.float32, 
        progress_weight: float, 
        distance_weight: float
        ) -> tuple[np.float32, np.float32, np.float32, np.float32]:

    frame_center = frame.Centroid
    nearest_idx, nearest_distance = CCP(frame_center, guide_curve)

    d_reward = distance_reward(nearest_distance, distance_weight)
    p_reward, new_max_t = progression_reward(nearest_idx, max_t, guide_curve, progress_weight)

    total_reward = d_reward + p_reward

    return total_reward, new_max_t, d_reward, p_reward

# =============================================================================
# FUNCTIONS - TERMINATION
# =============================================================================

def check_termination(
        frame              : BikeFrame,
        curve_end          : np.ndarray,
        curve_end_tangent  : np.ndarray,
        current_step       : int,
        max_step           : int,
        strict_termination : bool,
    ) -> tuple[bool, float]:

    steps_remaining = max_step - current_step
    centroid        = frame.Centroid

    within_vicinity = distance_to(centroid, curve_end) < IV.termination_vicinity
    overshot        = dot(centroid - curve_end, curve_end_tangent) > 0.0

    if strict_termination:
        if overshot:
            return True, IV.overshot_penalty
        if within_vicinity:
            return True, steps_remaining * IV.termination_step
    else:
        if within_vicinity:
            return True, steps_remaining * IV.termination_step
        if overshot:
            return True, IV.overshot_penalty

    return False, 0.0

# =============================================================================
# FUNCTIONS - RENDERING
# =============================================================================

def coordinate_to_pixel(point, window_size, bounds, bounding_range):
    pixel_x = int((point[0] - bounds["x_min"]) * window_size[0] / bounding_range[0])
    pixel_z = window_size[1] - int((point[1] - bounds["z_min"]) * window_size[1] / bounding_range[1]) # Pygame builds the z-axis downwards
    pixel_coordinate = (pixel_x, pixel_z)

    return pixel_coordinate

