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

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ShapeGrammar:
    size   : int
    labels : list[str] | None = None
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
    Connection Log: Target, Candidate, Mirror
    '''
    def __init__(self, placed_frames: list[BikeFrame], connection_log: list[tuple[PointDict, PointDict, bool]]):
        self.placed_frames  = placed_frames
        self.connection_log = connection_log

    # ─────────────────────────────────────────────────────────────────────
    # SUPPORTS - simply lowest point of first and last frame.
    # ─────────────────────────────────────────────────────────────────────
    @cached_property
    def pin(self) -> np.ndarray:
        first_joints = self.placed_frames[0].points
        lowest_idx   = np.argmin(first_joints[:,1])
        return first_joints[lowest_idx]

    @cached_property
    def roller(self) -> np.ndarray:
        last_joints = self.placed_frames[-1].points
        lowest_idx  = np.argmin(last_joints[:,1])
        return last_joints[lowest_idx]

    # ─────────────────────────────────────────────────────────────────────
    # CONNECTIONS
    # ─────────────────────────────────────────────────────────────────────
    @cached_property
    def connections(self):
        target_lines    = []
        candidate_lines = []

        for i, (target, candidate, mirror) in enumerate(self.connection_log):
            if i == 0:
                continue   # +++ first frame is not relevant for connections

            target_frame    = self.placed_frames[i - 1]
            candidate_frame = self.placed_frames[i]

            ta, tb = Pairs[target.value]
            ca, cb = Pairs[candidate.value]

            target_lines.append((target_frame.points[ta], target_frame.points[tb]))
            candidate_lines.append((candidate_frame.points[ca], candidate_frame.points[cb]))

        connection_sets = []

        for i, ((target_a, target_b), (candidate_a, candidate_b)) in enumerate(
            zip(target_lines, candidate_lines), start=1
        ):
            target_mirror    = self.connection_log[i - 1][2]
            candidate_mirror = self.connection_log[i][2]

            if target_mirror != candidate_mirror:
                connection_sets.append([
                    [target_a, candidate_a],
                    [target_b, candidate_b],
                ])
            else:
                connection_sets.append([
                    [target_a, candidate_b],
                    [target_b, candidate_a],
                ])

        long_side  = []
        short_side = []
        for connection in connection_sets:
            [t_a, c_a], [t_b, c_b] = connection

            target_len    = np.linalg.norm(t_b - t_a)
            candidate_len = np.linalg.norm(c_b - c_a)

            if target_len >= candidate_len:
                long_side.append([t_a, t_b])
                short_side.append([c_a, c_b])
            else:
                long_side.append([c_a, c_b])
                short_side.append([t_a, t_b])

        perpendicular_points = []
        for (long_a, long_b), (short_a, short_b) in zip(long_side, short_side):
            ab    = long_b - long_a
            ab_sq = np.dot(ab, ab)

            projected = []
            for short_pt in (short_a, short_b):
                t = np.dot(short_pt - long_a, ab) / ab_sq
                assert 0.0 <= t <= 1.0, (
                    f"Perpendicular projection fell outside long line segment (t={t:.4f}). "
                    f"This should be geometrically impossible — check upstream logic."
                )
                foot = long_a + t * ab
                projected.append(foot)

            perpendicular_points.append(projected)

        return connection_sets

