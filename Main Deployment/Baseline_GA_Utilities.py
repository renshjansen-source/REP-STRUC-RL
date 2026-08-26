''' This file contains the utility functions specific to the baseline GA reconstruction.

These functions build on top of the RL framework's BikeBuilder_Classes and
BikeBuilder_Utilities without modifying or reimplementing anything from those
files. Placement math, reward math, and intersection checks are all reused as-is.
'''

# =============================================================================
# IMPORTS
# =============================================================================

from dataclasses import dataclass
from itertools import product
from typing import Optional

import numpy as np

from internal_variables import IV
from environment.envs.BikeBuilder_Classes import BikeFrame, PointDict, Plane
from environment.envs.BikeBuilder_Utilities import (
    place_first,
    place,
    step_reward,
    frames_intersect_proximity,
)

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RankedCandidate:
    orientation  : tuple[PointDict, PointDict, bool]   # (target, candidate, mirror)
    placed_frame : BikeFrame
    total_reward : np.float32
    d_reward     : np.float32
    p_reward     : np.float32
    new_max_t    : np.float32

# =============================================================================
# ORIENTATION ENUMERATION
# =============================================================================

def enumerate_orientations() -> list[tuple[PointDict, PointDict, bool]]:
    # All 50 (target, candidate, mirror) combinations, fixed order so ranking
    # and rank-index selection are reproducible across runs.
    orientations = []
    for target, candidate, mirror in product(PointDict, PointDict, (False, True)):
        orientations.append((target, candidate, mirror))
    return orientations

# =============================================================================
# PLACEMENT DISPATCH
# =============================================================================

def place_candidate(
        frame          : BikeFrame,
        orientation    : tuple[PointDict, PointDict, bool],
        t              : int,
        mirror_flag    : bool,
        initial_planes : Optional[list[Plane]] = None,
        previous_frame : Optional[BikeFrame] = None,
        ) -> BikeFrame:
    target, candidate, mirror = orientation

    if t == 0:
        assert initial_planes is not None, (
            "initial_planes must be provided for the first-frame placement (t == 0)."
        )
        return place_first(frame, initial_planes, candidate, target, mirror)

    assert previous_frame is not None, (
        "previous_frame must be provided for non-first-frame placement (t > 0)."
    )
    return place(frame, previous_frame, candidate, target, mirror, mirror_flag)

# =============================================================================
# RANKING
# =============================================================================

def rank_candidates(
        frame          : BikeFrame,
        t              : int,
        mirror_flag    : bool,
        guide_curve    : np.ndarray,
        max_t          : np.float32,
        progress_weight: float,
        distance_weight: float,
        initial_planes : Optional[list[Plane]] = None,
        previous_frame : Optional[BikeFrame] = None,
        ) -> list[RankedCandidate]:
    # Places all 50 orientations for the picked frame, computes step_reward for
    # each, keeps only genuine-progress candidates, and sorts descending by
    # total_reward. step_reward is reused wholesale (not split into a lighter
    # progress-only check) per the agreed approach.

    orientations = enumerate_orientations()
    survivors: list[RankedCandidate] = []

    for orientation in orientations:
        placed_frame = place_candidate(
            frame          = frame,
            orientation    = orientation,
            t              = t,
            mirror_flag    = mirror_flag,
            initial_planes = initial_planes,
            previous_frame = previous_frame,
        )

        total_reward, new_max_t, d_reward, p_reward = step_reward(
            placed_frame, guide_curve, max_t, progress_weight, distance_weight
        )

        if p_reward == IV.no_progress:
            continue

        survivors.append(RankedCandidate(
            orientation  = orientation,
            placed_frame = placed_frame,
            total_reward = total_reward,
            d_reward     = d_reward,
            p_reward     = p_reward,
            new_max_t    = new_max_t,
        ))

    survivors.sort(key=lambda candidate: candidate.total_reward, reverse=True)
    return survivors

# =============================================================================
# INTERSECTION RESOLUTION
# =============================================================================

def resolve_placement(
        ranked_candidates  : list[RankedCandidate],
        picked_orientation : int,
        rank_window        : int,
        placed_frames      : list[BikeFrame],
        ) -> tuple[Optional[RankedCandidate], str, Optional[int]]:
    '''
    Resolves the final placement from a progress-ranked candidate list.

    Returns (chosen_candidate_or_None, outcome, used_rank):
      outcome is one of "placed", "no_progress", "no_intersection".
      chosen_candidate is the full RankedCandidate (frame + orientation +
      reward fields), not just the placed frame, so the caller never needs to
      re-derive which orientation was actually selected.
      used_rank is the index into the final clean-candidate list that was
      selected (only meaningful when outcome == "placed").

    Behaviour:
      - 0 progress-survivors                    -> ("no_progress" skip)
      - top rank_window checked left-to-right (highest reward first); a
        candidate that intersects is culled, everything below it shifts up,
        and the next-best not-yet-included candidate is appended at the
        BOTTOM of the window (not into the vacated slot). The walk continues
        and will check that appended candidate in its own turn, so failures
        can cascade further down the full ranked list without a separate
        pre-check.
      - 0 intersection-clean survivors           -> ("no_intersection" skip)
      - 1+ clean survivors, fewer than requested -> wrap-select via modulo
      - enough clean survivors                   -> direct index
    '''
    if len(ranked_candidates) == 0:
        return None, "no_progress", None

    window = list(ranked_candidates[:rank_window])
    backfill_pool = list(ranked_candidates[rank_window:])  # rank 9, 10, ... in original order

    clean: list[RankedCandidate] = []
    idx = 0
    while idx < len(window):
        candidate = window[idx]
        if frames_intersect_proximity(candidate.placed_frame, placed_frames, IV.intersect_buffer):
            # Cull this candidate. Everything below shifts up by one (implicit,
            # since we pop at idx), and the next-best not-yet-included candidate
            # is appended at the END of the window — not into the gap. It will
            # be checked for intersection in its own right when the walk
            # reaches it, so no separate pre-check is needed here.
            window.pop(idx)
            if backfill_pool:
                window.append(backfill_pool.pop(0))
            # If backfill_pool is exhausted, window just stays one shorter.
            continue  # re-check whatever now sits at idx (shifted up from below)
        clean.append(candidate)
        idx += 1

    if len(clean) == 0:
        return None, "no_intersection", None

    used_rank = picked_orientation % len(clean)
    chosen    = clean[used_rank]
    return chosen, "placed", used_rank

# =============================================================================
# INPUT CONVERSION (pure — no Rhino types)
# =============================================================================

def frame_points_from_lines(line_endpoints: list[tuple[tuple[float, float], tuple[float, float]]]) -> np.ndarray:
    '''
    Converts one stock frame's 5 lines (each given as ((x0,z0), (x1,z1)) point
    pairs, in ST_TOP-TT-HT-DT-CS_SS winding order matching the old GA's line
    order) into the 5 raw corner points BikeFrame expects.

    Line order (matching Pairs / old connection_names winding):
      0: TT  (ST_TOP -> HT_TOP)
      1: HT  (HT_TOP -> HT_BOTTOM)
      2: DT  (HT_BOTTOM -> BB)
      3: CS  (BB -> CS_SS)
      4: SS  (CS_SS -> ST_TOP)

    Each line's start point supplies one corner; line 0's start is ST_TOP,
    line 1's start is HT_TOP, and so on — i.e. corner i = line i's start point.
    This mirrors how the old scripts read `line.From` for each of the 5 lines
    to reconstruct the frame's corner points.
    '''
    assert len(line_endpoints) == 5, (
        f"Expected exactly 5 lines per frame, got {len(line_endpoints)}."
    )
    corners = np.array([start for (start, _end) in line_endpoints], dtype=np.float32)
    return corners

def guide_curve_from_points(curve_points: list[tuple[float, float]]) -> np.ndarray:
    # Converts a flat list of (x, z) tuples (already ordered start -> end
    # along the curve) into the np.ndarray shape resample_curve expects.
    return np.array(curve_points, dtype=np.float32)

# =============================================================================
# OUTPUT CONVERSION (pure — no Rhino types)
# =============================================================================

def frame_to_point_tuples(frame: BikeFrame) -> list[tuple[float, float]]:
    # Edge points only (the 5 raw corners), as plain (x, z) tuples — the GH
    # script wraps each in rg.Point3d(x, 0, z) or similar.
    return [(float(p[0]), float(p[1])) for p in frame.points]

def placed_frames_to_point_tuples(placed_frames: list[BikeFrame]) -> list[list[tuple[float, float]]]:
    return [frame_to_point_tuples(frame) for frame in placed_frames]

def bridge_points_to_tuples(bike_bridge) -> list[list[tuple[float, float]]]:
    # bike_bridge.points is already a list (per frame) of lists of np.ndarray
    # points (raw corners plus any inserted connection points). Converts every
    # point to a plain (x, z) tuple, preserving the per-frame nesting so the
    # GH script can build one Point3d branch per frame.
    return [
        [(float(p[0]), float(p[1])) for p in frame_points]
        for frame_points in bike_bridge.points
    ]