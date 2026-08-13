''' This file contains assisting class functions for the BikeBuilder Environment file '''

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
from numpy.typing import NDArray
from enum import Enum
from internal_variables import IV
from dataclasses import dataclass, field

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
