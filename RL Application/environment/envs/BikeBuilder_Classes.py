''' This file contains assisting class functions for the BikeBuilder Environment file '''

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
from numpy.typing import NDArray
from enum import Enum
from internal_variables import IV
from dataclasses import dataclass, field
from functools import cached_property
from collections import defaultdict # Allows for reading of keys which don't exist yet
from typing import Optional # Required for GH-level access to functions, replaces type | None = None type assertions. (GH python is older)

# =============================================================================
# TYPE ALIASES
# =============================================================================

Vector2D = NDArray[np.float32]
Point2D  = NDArray[np.float32]

# =============================================================================
# PLAIN DICTIONARIES
# =============================================================================
Pairs = {
    0: (0, 1),   # TT_MID : ST_TOP  → HT_TOP
    1: (1, 2),   # HT_MID : HT_TOP  → HT_BOTTOM
    2: (2, 3),   # DT_MID : HT_BOTTOM → BB
    3: (3, 4),   # CS_MID : BB      → CS_SS
    4: (4, 0),   # SS_MID : CS_SS   → ST_TOP
}

OUTER_TUBE_PROPERTIES = ("top_tubes", "head_tubes", "down_tubes", "chain_stays", "seat_stays")

# =============================================================================
# ASSISTING FUNCTIONS
# =============================================================================
def cross(vector_a: Vector2D, vector_b: Vector2D) -> float:
    return vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]

def dot(vector_a: Vector2D, vector_b: Vector2D) -> float:
    return vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]

def segments_intersect(p1, p2, p3, p4) -> bool:
    d1 = p2 - p1
    d2 = p4 - p3
    cross_product = cross(d1, d2)

    if abs(cross_product) < IV.intersect_tol:
        return False # Guarding against 0 division

    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross_product
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross_product

    return 0.0 < t < 1.0 and 0.0 < u < 1.0
# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ShapeGrammar:
    size   : int
    labels : Optional[list[str]] = None
    counts : np.ndarray = field(init = False)

    def __post_init__(self):
        self.counts = np.zeros(self.size, dtype=np.int64)
        if self.labels is not None and len(self.labels) != self.size:
            raise ValueError(f"labels length {len(self.labels)} does not match size {self.size}")

    def record(self, index: int) -> None:
        self.counts[index] += 1

    def reset(self) -> None:
        self.counts[:] = 0                      # [:] = entire array

    def as_dict(self) -> dict:
        if self.labels is not None:
            return {label: int(c) for label, c in zip(self.labels, self.counts)}
        return{str(i): int(c) for i, c in enumerate(self.counts)}

@dataclass
class EpisodeGrammar:
    stock     : ShapeGrammar
    target    : ShapeGrammar
    candidate : ShapeGrammar
    mirror    : ShapeGrammar

    def record(self, action) -> None:
        self.stock.record(action[0])
        self.target.record(action[1])
        self.candidate.record(action[2])
        self.mirror.record(action[3])

    def reset(self) -> None:
        self.stock.reset()
        self.target.reset()
        self.candidate.reset()
        self.mirror.reset()
    
# =============================================================================
# CLASSES
# =============================================================================
class Plane:
    def __init__(self, origin: np.ndarray, x_vector: np.ndarray, mirror: bool = False):
        self.Origin   = origin
        self.X_vector = x_vector
        self.mirror   = mirror

    @property
    def Y_vector(self) -> np.ndarray:
        if not self.mirror:
            return np.array([-self.X_vector[1], self.X_vector[0]], dtype=np.float32)
        else:
            return np.array([self.X_vector[1], -self.X_vector[0]], dtype=np.float32)

class PointDict(Enum):
    TT_MID = 0  # midpoint(ST_TOP, HT_TOP)
    HT_MID = 1  # midpoint(HT_TOP, HT_BOTTOM)
    DT_MID = 2  # midpoint(HT_BOTTOM, BB)
    CS_MID = 3  # midpoint(BB, CS_SS)
    SS_MID = 4  # midpoint(CS_SS, ST_TOP)

class BikeFrame:
    def __init__(self, points: np.ndarray, recenter: bool = True):
        inspace_positions = np.array(points, dtype = np.float32)
        if recenter:
            bb_base_point = inspace_positions[3]
            self.points   = inspace_positions - bb_base_point
        else:
            self.points   = inspace_positions
    
    @property
    def mid_points(self) -> np.ndarray:
        # Building the attachment points
        mids = np.array([
            (self.points[0] + self.points[1]) / 2, 
            (self.points[1] + self.points[2]) / 2,  
            (self.points[2] + self.points[3]) / 2,  
            (self.points[3] + self.points[4]) / 2,  
            (self.points[4] + self.points[0]) / 2,  
        ], dtype=np.float32)
    
        return mids

    @property
    def mid_vectors(self) -> np.ndarray:
        # Building the attachment vectors
        mids = np.array([
            self.points[1] - self.points[0],  
            self.points[2] - self.points[1],  
            self.points[3] - self.points[2],  
            self.points[4] - self.points[3],  
            self.points[0] - self.points[4],  
        ], dtype=np.float32)
        vec_lengths = np.linalg.norm(mids, axis=1, keepdims=True)

        return mids / vec_lengths

    @property
    def observation_points(self) -> np.ndarray:
        # returns point-midpoint... to be used in the obs
        interleaved = np.empty((10, 2), dtype=np.float32)
        interleaved[0::2] = self.points
        interleaved[1::2] = self.mid_points
        return interleaved

    @property
    def Centroid(self) -> np.ndarray:
        # Retrieves the centroid on the frame
        center_point = np.mean(self.points, axis=0)
        return center_point

    @property
    def turning_angles(self) -> np.ndarray:
        edges  = np.roll(self.points, -1, axis=0) - self.points
        e_prev = np.roll(edges, 1, axis=0)
        cross_vals = e_prev[:, 0] * edges[:, 1] - e_prev[:, 1] * edges[:, 0]
        dot_vals   = e_prev[:, 0] * edges[:, 0] + e_prev[:, 1] * edges[:, 1]
        return np.arctan2(cross_vals, dot_vals).astype(np.float32)

    def mid_pt(self, index: PointDict):
        # For retrieving only a single mid point
        idx_a, idx_b = Pairs[index.value]
        point_a      = self.points[idx_a]
        point_b      = self.points[idx_b]
        mid          = (point_a + point_b) / 2

        return mid

    def mid_vec(self, index: PointDict):
        # For retrieving a single vector
        idx_a, idx_b = Pairs[index.value]
        point_a      = self.points[idx_a]
        point_b      = self.points[idx_b]
        vec          = point_b - point_a

        return vec
        

    def get_candidate_plane(self, index: PointDict, mirror: bool):
        # Retrieving Corner Points
        idx_a, idx_b = Pairs[index.value]
        point_a      = self.points[idx_a]
        point_b      = self.points[idx_b]

        # Building Plane
        mid     = (point_a + point_b) / 2
        tangent = point_a - point_b
        
        return Plane(mid, tangent, mirror=mirror)

    def get_target_plane(self, index: PointDict, mirror_flag):
        # Retrieving Corner Points
        idx_a, idx_b = Pairs[index.value]
        point_a      = self.points[idx_a]
        point_b      = self.points[idx_b]

        # Building Plane
        mid     = (point_a + point_b) / 2
        tangent = point_a - point_b

        x_vector = tangent if mirror_flag else -tangent

        return Plane(mid, x_vector, mirror=False) # Temporary revised

class BikeBridge:
    '''
    Class to be used for the FEA side of things
    Connection Log: Stock Index, Target, Candidate, Mirror
    '''
    def __init__(self, placed_frames: list[BikeFrame], connection_log: list[tuple[int, PointDict, PointDict, bool]]):
        self.placed_frames  = placed_frames
        self.connection_log = connection_log
        self.stock_indices  = [entry[0] for entry in connection_log]

        self.points, self.corner_index, self.perpendicular_lookup = self.build_points()

    def build_points(self):
        n_frames = len(self.placed_frames)
        insertions: dict[tuple[int, int], list[tuple[float, np.ndarray, int, int]]] = defaultdict(list)
        perpendicular_lookup: dict[tuple[int, int], tuple[int, int] | None] = {}

        for i in range(1, n_frames):
            _, target, candidate, _ = self.connection_log[i]

            target_frame    = self.placed_frames[i - 1]
            candidate_frame = self.placed_frames[i]

            ta, tb = Pairs[target.value]
            ca, cb = Pairs[candidate.value]

            target_a, target_b       = target_frame.points[ta],    target_frame.points[tb]
            candidate_a, candidate_b = candidate_frame.points[ca], candidate_frame.points[cb]

            target_len    = np.linalg.norm(target_b - target_a)
            candidate_len = np.linalg.norm(candidate_b - candidate_a)

            if target_len >= candidate_len:
                long_frame_idx, long_edge_id = i - 1, target.value
                long_a, long_b               = target_a, target_b
                short_a, short_b             = candidate_a, candidate_b
            else:
                long_frame_idx, long_edge_id = i, candidate.value
                long_a, long_b               = candidate_a, candidate_b
                short_a, short_b             = target_a, target_b

            ab    = long_b - long_a
            ab_sq = np.dot(ab, ab)

            for slot, short_pt in enumerate((short_a, short_b)):
                raw_t = np.dot(short_pt - long_a, ab) / ab_sq
                t     = float(np.clip(raw_t, 0.0, 1.0))
                foot  = long_a + t * ab

                nearer_corner = long_a if t <= 0.5 else long_b
                distance_to_corner = float(np.linalg.norm(foot - nearer_corner))

                if distance_to_corner < IV.minimum_connection_distance:
                    perpendicular_lookup[(i, slot)] = None
                    continue

                insertions[(long_frame_idx, long_edge_id)].append((t, foot, i, slot))

        points       : list[list[np.ndarray]] = []
        corner_index : list[list[int]]        = []

        for frame_idx, frame in enumerate(self.placed_frames):
            frame_points       = []
            frame_corner_index = [0, 0, 0, 0, 0]

            for corner_id in range(5):
                frame_corner_index[corner_id] = len(frame_points)
                frame_points.append(frame.points[corner_id])

                edge_id = corner_id
                edge_insertions = sorted(insertions.get((frame_idx, edge_id), []), key=lambda entry: entry[0])
                for t, point, conn_i, slot in edge_insertions:
                    local_idx = len(frame_points)
                    frame_points.append(point)
                    perpendicular_lookup[(conn_i, slot)] = (frame_idx, local_idx)

            points.append(frame_points)
            corner_index.append(frame_corner_index)

        return points, corner_index, perpendicular_lookup

    def tube_indices(self, edge_id: int) -> list[list[int]]:
        corner_a, corner_b = Pairs[edge_id]

        if corner_a < corner_b:
            return [
                list(range(frame_map[corner_a], frame_map[corner_b] + 1))
                for frame_map in self.corner_index
            ]

        # Wraparound case — only edge 4 (SS_MID, corners 4 -> 0)
        result = []
        for frame_idx, frame_map in enumerate(self.corner_index):
            n_points = len(self.points[frame_idx])
            indices  = list(range(frame_map[corner_a], n_points)) + [frame_map[corner_b]]
            result.append(indices)
        return result

    @cached_property
    def top_tubes(self) -> list[list[int]]:
        return self.tube_indices(0)

    @cached_property
    def head_tubes(self) -> list[list[int]]:
        return self.tube_indices(1)

    @cached_property
    def down_tubes(self) -> list[list[int]]:
        return self.tube_indices(2)

    @cached_property
    def chain_stays(self) -> list[list[int]]:
        return self.tube_indices(3)

    @cached_property
    def seat_stays(self) -> list[list[int]]:
        return self.tube_indices(4)

    @cached_property
    def seat_tubes(self) -> list[list[int]]:
        return [[frame_map[3], frame_map[0]] for frame_map in self.corner_index] 

    @cached_property
    def connections(self):
        n_frames = len(self.placed_frames)
        connection_sets = []

        for i in range(1, n_frames):
            _, target, candidate, _ = self.connection_log[i]

            target_frame_idx    = i - 1
            candidate_frame_idx = i

            ta, tb = Pairs[target.value]
            ca, cb = Pairs[candidate.value]

            target_a_id    = (target_frame_idx,    self.corner_index[target_frame_idx][ta])
            target_b_id    = (target_frame_idx,    self.corner_index[target_frame_idx][tb])
            candidate_a_id = (candidate_frame_idx, self.corner_index[candidate_frame_idx][ca])
            candidate_b_id = (candidate_frame_idx, self.corner_index[candidate_frame_idx][cb])

            target_frame_obj    = self.placed_frames[target_frame_idx]
            candidate_frame_obj = self.placed_frames[candidate_frame_idx]
            target_len    = np.linalg.norm(target_frame_obj.points[tb]    - target_frame_obj.points[ta])
            candidate_len = np.linalg.norm(candidate_frame_obj.points[cb] - candidate_frame_obj.points[ca])
            target_is_long = target_len >= candidate_len

            # Mirrors build_points' short_a/short_b assignment, so slot identity resolves consistently
            if target_is_long:
                short_a_id, short_b_id = candidate_a_id, candidate_b_id
            else:
                short_a_id, short_b_id = target_a_id, target_b_id

            target_mirror    = self.connection_log[i - 1][3]
            candidate_mirror = self.connection_log[i][3]

            if target_mirror == candidate_mirror:
                pair_1 = (target_a_id, candidate_b_id)
                pair_2 = (target_b_id, candidate_a_id)
            else:
                pair_1 = (target_a_id, candidate_a_id)
                pair_2 = (target_b_id, candidate_b_id)

            entries = []
            for target_side_id, candidate_side_id in (pair_1, pair_2):
                long_id, short_id = (target_side_id, candidate_side_id) if target_is_long \
                                    else (candidate_side_id, target_side_id)

                assert short_id == short_a_id or short_id == short_b_id, (
                    f"short_id {short_id} matches neither short_a_id nor short_b_id "
                    f"for connection {i} — pairing logic is inconsistent."
                )
                slot    = 0 if short_id == short_a_id else 1
                perp_id = self.perpendicular_lookup.get((i, slot))

                if perp_id is None:
                    entries.append([long_id, short_id])
                    continue

                long_point        = self.points[long_id[0]][long_id[1]]
                short_point       = self.points[short_id[0]][short_id[1]]
                end_point_length  = np.linalg.norm(long_point - short_point)

                if IV.enable_connection_limit and end_point_length > IV.connection_limit:
                    entries.append([short_id, perp_id])
                else:
                    entries.append([long_id, short_id, perp_id])

            connection_sets.append(entries)

        return connection_sets
    
    @cached_property
    def pin(self):
        first_frame_points = np.array(self.points[0])
        local_idx = int(np.argmin(first_frame_points[:, 1]))
        return (0, local_idx)

    @cached_property
    def roller(self):
        last_frame_idx    = len(self.points) - 1
        last_frame_points = np.array(self.points[last_frame_idx])
        local_idx = int(np.argmin(last_frame_points[:, 1]))
        return (last_frame_idx, local_idx)  
    
    @cached_property
    def load_data(self):
        n_frames = len(self.placed_frames)

        raw_candidates = []
        for frame_idx in range(n_frames):
            for corner_id in (0, 3):
                local_idx = self.corner_index[frame_idx][corner_id]
                point     = self.points[frame_idx][local_idx]
                raw_candidates.append(((frame_idx, local_idx), point))

        tube_index_lists_per_frame = [[] for _ in range(n_frames)]
        for prop_name in OUTER_TUBE_PROPERTIES:
            per_frame_indices = getattr(self, prop_name)
            for frame_idx, index_list in enumerate(per_frame_indices):
                tube_index_lists_per_frame[frame_idx].append(index_list)

        deck_min, deck_max = IV.deck_range

        valid_candidates = []
        for (frame_idx, local_idx), point in raw_candidates:
            if not (deck_min <= point[0] <= deck_max):
                continue # Filters out candidates which exceed the deck_range

            ray_origin = point + np.array([0.0, IV.connection_offset / 2.0], dtype=np.float32)
            ray_end    = ray_origin + np.array([0.0, IV.ray_height], dtype=np.float32)

            window_start = max(0, frame_idx - 2)
            window_end   = min(n_frames - 1, frame_idx + 2)

            blocked = False
            for window_frame_idx in range(window_start, window_end + 1):
                for tube_index_list in tube_index_lists_per_frame[window_frame_idx]:
                    for k in range(len(tube_index_list) - 1):
                        seg_a = self.points[window_frame_idx][tube_index_list[k]]
                        seg_b = self.points[window_frame_idx][tube_index_list[k + 1]]
                        if segments_intersect(ray_origin, ray_end, seg_a, seg_b):
                            blocked = True
                            break
                    if blocked:
                        break
                if blocked:
                    break

            if not blocked:
                valid_candidates.append(((frame_idx, local_idx), point))

        if len(valid_candidates) < IV.load_divider:
            return [], False

        xs = np.array([point[0] for _, point in valid_candidates], dtype=np.float32)
        x_min, x_max = float(xs.min()), float(xs.max())
        assert x_max > x_min, (
            "All valid load candidates share the same X value — cannot spread selection."
        )

        normalized_xs = (xs - x_min) / (x_max - x_min)
        targets       = np.linspace(0.0, 1.0, IV.load_divider)

        picked   = set()
        selected = []
        for target in targets:
            distances = np.abs(normalized_xs - target)
            for idx in picked:
                distances[idx] = np.inf
            best = int(np.argmin(distances))
            picked.add(best)
            selected.append(valid_candidates[best][0])

        return selected, True

    @cached_property
    def load_points(self):
        ids, _ = self.load_data
        return ids

    @cached_property
    def load_valid(self):
        _, valid = self.load_data
        return valid

    @cached_property
    def tension_data(self):
        # Step 1 — one candidate per frame: its lowest (min z) raw corner
        candidates = []  # list of (id, point) pairs, one per frame
        for frame_idx, frame in enumerate(self.placed_frames):
            corner_id = int(np.argmin(frame.points[:, 1]))
            local_idx = self.corner_index[frame_idx][corner_id]
            point     = self.points[frame_idx][local_idx]
            candidates.append(((frame_idx, local_idx), point))

        # Step 2 — sort left to right by X
        candidates.sort(key=lambda entry: entry[1][0])

        # Step 3 — bail out if fewer than 2 candidates
        if len(candidates) < 2:
            return [], False

        # Step 4 — greedy chord building
        line_out = []
        i = 0
        while i < len(candidates):
            current_id, current_pt = candidates[i]
            end_idxs = list(range(i + 1, min(i + 1 + IV.tension_count, len(candidates))))

            if not end_idxs:
                break

            chord_options = []
            for j in end_idxs:
                end_id, end_pt = candidates[j]
                length = float(np.linalg.norm(end_pt - current_pt))
                chord_options.append((j, end_id, end_pt, length))

            low, high = IV.tension_thresh
            chord_options = [option for option in chord_options if low <= option[3] <= high]

            if not chord_options:
                i += 1
                continue

            # Intersection filter — trim ends, check against every frame's 5 raw corner-to-corner edges
            valid_options = []
            for j, end_id, end_pt, length in chord_options:
                direction     = (end_pt - current_pt) / length
                trimmed_start = current_pt + IV.tension_trim * direction
                trimmed_end   = end_pt     - IV.tension_trim * direction

                intersects = False
                for frame in self.placed_frames:
                    for corner_a, corner_b in Pairs.values():
                        if segments_intersect(
                            trimmed_start, trimmed_end,
                            frame.points[corner_a], frame.points[corner_b]
                        ):
                            intersects = True
                            break
                    if intersects:
                        break

                if not intersects:
                    valid_options.append((j, end_id, end_pt, length))

            if not valid_options:
                i += 1
                continue

            best_j, best_end_id, _, _ = max(valid_options, key=lambda option: option[3])
            line_out.append((current_id, best_end_id))
            i = best_j

        tension_success = len(line_out) > 0
        return line_out, tension_success

    @cached_property
    def tension_lines(self):
        lines, _ = self.tension_data
        return lines

    @cached_property
    def tension_valid(self):
        _, valid = self.tension_data
        return valid