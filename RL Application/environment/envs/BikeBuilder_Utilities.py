''' This file contains the utility functions '''

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import math
from scipy.interpolate import interp1d

from internal_variables import IV
from environment.envs.BikeBuilder_Classes import Plane, BikeFrame, PointDict, Pairs, Vector2D, Point2D, ShapeGrammar, cross, dot, segments_intersect

# =============================================================================
# FUNCTIONS - MATHMATICAL FUNCTIONS
# =============================================================================
def remap(value: float, old_bounds: tuple[float, float], new_bounds: tuple[float, float]) -> float:
    old_min, old_max = old_bounds
    new_min, new_max = new_bounds

    t = (value - old_min) / (old_max - old_min)
    return new_min + t * (new_max - new_min)

def recip(value: float):
    return 1 / (math.sqrt(value))

# =============================================================================
# FUNCTIONS - VECTOR AND POINT MANIPULATION
# =============================================================================
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

def build_observation_points_positive(frame_stock: list, obs_type: str) -> np.ndarray:
    # Determines a shared [x_min,x_max] x [z_min,z_max] box from the ENTIRE input
    # stock, then remaps every frame's points into that box, independently per
    # axis, into [0,1]. No norm_range is passed in -- it's derived here.
    if obs_type == "combined":
        pts_fn = lambda f: f.observation_points
    elif obs_type in ("edge", "angle"):
        pts_fn = lambda f: f.points
    elif obs_type == "mid":
        pts_fn = lambda f: f.mid_points
    else:
        raise ValueError(f"Unknown obs_type {obs_type!r} - expected one of 'combined', 'edge', 'mid', 'angle'")

    raw_stock   = np.stack([pts_fn(frame) for frame in frame_stock])   # (n_frames, n_points, 2)
    stock_min   = raw_stock.min(axis=(0, 1))                           # [x_min, z_min], whole stock
    stock_max   = raw_stock.max(axis=(0, 1))                           # [x_max, z_max], whole stock
    stock_range = stock_max - stock_min                                # independent per-axis span

    normalized_stock = []
    for frame in frame_stock:
        pts = (pts_fn(frame) - stock_min) / stock_range                # -> [0,1] per axis, independently
        if obs_type == "angle":
            angle_encoded = encode_angles(frame.turning_angles)        # never touched by this remap
            n = len(pts)
            pts = np.stack([pts, angle_encoded], axis=1).reshape(n * 2, 2)
        normalized_stock.append(pts.astype(np.float32))

    return np.array(normalized_stock, dtype=np.float32)

# =============================================================================
# FUNCTIONS - PENALTIES
# =============================================================================
def nearest_frames(frame: BikeFrame, candidates: list[BikeFrame], n: int) -> list[BikeFrame]:
    if len(candidates) <= n:
        return candidates

    distances = [distance_to(frame.Centroid, other.Centroid) for other in candidates]
    order = np.argsort(distances)[:n]
    return [candidates[i] for i in order]

def frames_intersect_proximity(frame: BikeFrame, candidates: list[BikeFrame], n: int) -> bool:
    nearby = nearest_frames(frame, candidates, n)
    return frames_intersect(frame, nearby)

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
        full_overshot      : bool,
        terminal_reward_scale : bool,
    ) -> tuple[bool, float, bool]:

    steps_remaining = max_step - current_step
    centroid        = frame.Centroid

    distance        = distance_to(centroid, curve_end)
    within_vicinity = distance < IV.termination_vicinity
    is_overshot     = dot(centroid - curve_end, curve_end_tangent) > 0.0

    if terminal_reward_scale:
        reward = remap(distance, (IV.termination_vicinity, 0), IV.termination_scale)
    else:
        reward = IV.termination_step

    if strict_termination:
        if is_overshot:
            return True, IV.overshot_penalty, is_overshot
        if within_vicinity:
            return True, reward, is_overshot
    else:
        if within_vicinity:
            return True, reward, is_overshot
        if is_overshot and full_overshot:
            return True, IV.overshot_penalty, is_overshot
        

    return False, 0.0, False
# =============================================================================
# FUNCTIONS - RENDERING
# =============================================================================

def doubled_tube_section(outer_diameter, thickness):
    # Returns (D_new, t_new) for a single tube whose annulus area equals
    # 2x the input tube's area, while preserving the same inner diameter.
    inner_diameter     = outer_diameter - 2.0 * thickness
    new_outer_diameter = np.sqrt(2.0 * outer_diameter**2 - inner_diameter**2)
    new_thickness      = (new_outer_diameter - outer_diameter + 2.0 * thickness) / 2.0
    return new_outer_diameter, new_thickness

# =============================================================================
# FUNCTIONS - RENDERING
# =============================================================================

def coordinate_to_pixel(point, window_size, bounds, bounding_range):
    pixel_x = int((point[0] - bounds["x_min"]) * window_size[0] / bounding_range[0]) + IV.window_padding
    pixel_z = window_size[1] - int((point[1] - bounds["z_min"]) * window_size[1] / bounding_range[1]) + IV.window_padding # Pygame builds the z-axis downwards
    pixel_coordinate = (pixel_x, pixel_z)

    return pixel_coordinate

# =============================================================================
# FUNCTIONS - FEA REWARDS
# =============================================================================

def FEA_convergence_check(fea_result:dict) -> bool:
    if fea_result["converged"] == False:
        return False
    elif fea_result["max_displacement"] == None:
        return False
    elif fea_result["frame_stress"]["sig_max"] == None:
        return False
    elif fea_result["frame_stress"]["sig_min"] == None:
        return False
    elif fea_result["connector_stress"]["sig_max"] == None:
        return False
    elif fea_result["connector_stress"]["sig_min"] == None:
        return False
    elif fea_result["cable_stress"]["sig_max"] == None:
        return False
    elif fea_result["cable_stress"]["sig_min"] == None:
        return False
    else:
        return True


def exponential_reward(value: float, low: float, high: float, max_reward: float, steepness: float) -> float:
    assert high > low, f"exponential_reward requires high > low, got low={low}, high={high}"

    if value <= low:
        return max_reward
    if value >= high:
        return 0.0

    t = (value - low) / (high - low)
    return max_reward * math.exp(-steepness * t)

def fea_reward(fea_result: dict) -> tuple[float, float, float]:
    max_disp = fea_result["max_displacement"]
    sig_max  = fea_result["frame_stress"]["sig_max"]
    sig_min  = fea_result["frame_stress"]["sig_min"]

    deform_low, deform_high           = IV.deform_reward
    tension_low, tension_high         = IV.tension_reward
    compression_low, compression_high = IV.compression_reward

    if max_disp is None:
        deform_r = 0.0
    else:
        deform_r = exponential_reward(
            max_disp, deform_low, deform_high, IV.max_reward_deform, IV.fea_reward_steepness
        )

    if sig_max is None:
        tension_r = 0.0
    else:
        tension_magnitude = max(0.0, sig_max)   # negative sig_max means no tension present anywhere
        tension_r = exponential_reward(
            tension_magnitude, tension_low, tension_high, IV.max_reward_tension, IV.fea_reward_steepness
        )

    if sig_min is None:
        compression_r = 0.0
    else:
        compression_magnitude = max(0.0, -sig_min)   # positive sig_min means no compression present anywhere
        compression_r = exponential_reward(
            compression_magnitude, compression_low, compression_high, IV.max_reward_compression, IV.fea_reward_steepness
        )

    return deform_r, tension_r, compression_r

def recip_reward(value: float, value_range: tuple[float, float], reward_range: tuple[float, float]) -> float:
    if value <= value_range[0]:
        return reward_range[1]
    if value >= value_range[1]:
        return reward_range[0]

    bound_0        = recip(value_range[0])
    bound_1        = recip(value_range[1])
    adjusted_value = recip(value)

    return remap(adjusted_value, (bound_1, bound_0), reward_range)

def FEA_reward_recip(fea_result: dict) -> tuple[float, float, float]:
    # retrieve values
    max_disp = fea_result["max_displacement"]
    sig_max  = fea_result["frame_stress"]["sig_max"]
    sig_min  = -fea_result["frame_stress"]["sig_min"]

    # reward calculation
    disp_reward        = recip_reward(max_disp, IV.deform_reward,      (0, IV.max_reward_deform))
    tens_reward        = recip_reward(sig_max,  IV.tension_reward,     (0, IV.max_reward_tension))
    comp_reward        = recip_reward(sig_min,  IV.compression_reward, (0, IV.max_reward_compression))

    total_reward = (disp_reward, tens_reward, comp_reward)

    return total_reward