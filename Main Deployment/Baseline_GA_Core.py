''' This file contains the main orchestration loop for the baseline GA reconstruction.

Given a full sequence of frame picks and orientation-rank picks, this runs the
entire structure placement in one call (no Rhino/Grasshopper dependencies).
Reuses BikeBuilder_Classes and BikeBuilder_Utilities from the RL framework
directly; only the *selection* mechanism (rank + intersection resolution) is
new, living in Baseline_GA_Utilities.
'''

# =============================================================================
# IMPORTS
# =============================================================================

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from internal_variables import IV
from environment.envs.BikeBuilder_Classes import BikeFrame, PointDict, BikeBridge
from environment.envs.BikeBuilder_Utilities import initial_targets, check_termination

from Baseline_GA_Utilities import rank_candidates, resolve_placement

# =============================================================================
# INTERNAL CONSTANTS
# =============================================================================

RANK_WINDOW_DEFAULT = 8   # Tuned empirically — too large and the GA fails to converge.

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RunTally:
    reuse_skips        : int  = 0
    no_progress_skips  : int  = 0
    no_intersect_skips : int  = 0
    steps_completed     : int  = 0
    terminated          : bool = False
    overshot            : bool = False
    true_termination     : bool = False

@dataclass
class RunResult:
    placed_frames   : list[BikeFrame]
    connection_log  : list[tuple[PointDict, PointDict, bool]]
    bike_bridge     : Optional[BikeBridge]
    tally           : RunTally

# =============================================================================
# MAIN ORCHESTRATION FUNCTION
# =============================================================================

def run_baseline_ga(
        frame_stock       : list[BikeFrame],
        guide_curve       : np.ndarray,
        picked_frame       : list[int],
        picked_orientation : list[int],
        progress_weight     : float = 1.0,
        distance_weight     : float = 1.0,
        rank_window         : int   = RANK_WINDOW_DEFAULT,
        strict_termination  : bool  = False,
        ) -> RunResult:

    assert len(picked_frame) == len(picked_orientation), (
        f"picked_frame (len {len(picked_frame)}) and picked_orientation "
        f"(len {len(picked_orientation)}) must be the same length."
    )

    # --- State initialization, mirroring BikeBuilder_Env.reset() ---
    stock_mask     = np.ones(len(frame_stock), dtype=np.float32)
    placed_frames  : list[BikeFrame] = []
    connection_log : list[tuple[PointDict, PointDict, bool]] = []
    previous_frame : Optional[BikeFrame] = None
    mirror_flag    : bool = False
    max_t          : np.float32 = np.float32(0.0)

    tally = RunTally()

    curve_end        = guide_curve[-1]
    end_tangent      = guide_curve[-1] - guide_curve[-2]
    curve_end_tangent = end_tangent / np.linalg.norm(end_tangent)

    # --- Main loop ---
    for t, (frame_idx, orientation_pick) in enumerate(zip(picked_frame, picked_orientation)):

        # Reuse check — happens before any orientation generation, matching the RL env.
        if stock_mask[frame_idx] == 0.0:
            tally.reuse_skips += 1
            continue

        frame = frame_stock[frame_idx]

        # First-frame planes only need to be built once, at t == 0.
        initial_planes = initial_targets(frame, guide_curve) if t == 0 else None

        ranked = rank_candidates(
            frame           = frame,
            t               = t,
            initial_planes  = initial_planes,
            previous_frame  = previous_frame,
            mirror_flag     = mirror_flag,
            guide_curve     = guide_curve,
            max_t           = max_t,
            progress_weight = progress_weight,
            distance_weight = distance_weight,
        )

        chosen, outcome, _used_rank = resolve_placement(
            ranked_candidates  = ranked,
            picked_orientation = orientation_pick,
            rank_window        = rank_window,
            placed_frames      = placed_frames,
        )

        if outcome == "no_progress":
            tally.no_progress_skips += 1
            continue
        if outcome == "no_intersection":
            tally.no_intersect_skips += 1
            continue

        assert chosen is not None  # "placed" outcome always carries a candidate

        # Commit the placement using the exact orientation resolve_placement selected.
        placed_frame = chosen.placed_frame
        max_t        = chosen.new_max_t

        placed_frames.append(placed_frame)
        connection_log.append(chosen.orientation)
        previous_frame = placed_frame
        mirror_flag    = chosen.orientation[2]  # mirror bool
        stock_mask[frame_idx] = 0.0
        tally.steps_completed += 1

        # Termination check — runs after every committed placement.
        terminated, _terminal_reward, overshot = check_termination(
            placed_frame, curve_end, curve_end_tangent,
            tally.steps_completed, len(picked_frame), strict_termination
        )

        if terminated:
            tally.terminated      = True
            tally.overshot        = overshot
            tally.true_termination = terminated and not overshot
            break

    # --- BikeBridge construction, only on true termination ---
    bike_bridge = None
    if tally.true_termination:
        bike_bridge = BikeBridge(placed_frames, connection_log)

    return RunResult(
        placed_frames  = placed_frames,
        connection_log = connection_log,
        bike_bridge    = bike_bridge,
        tally          = tally,
    )