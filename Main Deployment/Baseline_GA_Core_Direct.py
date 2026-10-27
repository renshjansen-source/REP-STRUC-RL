''' Direct-action variant of the baseline GA reconstruction.

Identical in purpose to Baseline_GA_Core, with one difference: the GA supplies
target, candidate and mirror directly per step, instead of a frame pick plus an
orientation rank. Each step is therefore the same four-part action the RL env
receives: [frame, target, candidate, mirror].

There is no ranking, no rank window and no progress filter. Each step follows
BikeBuilder_Env.step():
  - reuse          -> skip (pick consumed, nothing placed)
  - first frame    -> place_first, no intersection or termination check
  - other frames   -> place, intersection check (frames_intersect_proximity),
                      skip on intersection
  - commit         -> termination check (not on the first frame)

RunTally / RunResult are imported from Baseline_GA_Core so the calling script's
outputs stay identical to the ranked variant. Fields with no meaning here:
  - tally.no_progress_skips -> always 0 (the RL env penalises, never skips)
  - remaining_options       -> always []
'''

# =============================================================================
# IMPORTS
# =============================================================================

from typing import Optional

import numpy as np

from internal_variables import IV
from environment.envs.BikeBuilder_Classes import BikeFrame, PointDict, BikeBridge
from environment.envs.BikeBuilder_Utilities import (
    initial_targets,
    check_termination,
    frames_intersect_proximity,
)

from Baseline_GA_Core import RunTally, RunResult
from Baseline_GA_Utilities import place_candidate

# =============================================================================
# MAIN ORCHESTRATION FUNCTION
# =============================================================================

def run_baseline_ga_direct(
        frame_stock           : list[BikeFrame],
        guide_curve           : np.ndarray,
        picked_frame          : list[int],
        picked_target         : list[int],
        picked_candidate      : list[int],
        picked_mirror         : list[int],
        strict_termination    : bool = False,
        full_overshot         : bool = True,
        terminal_reward_scale : bool = False,
        ) -> RunResult:

    n_picks = len(picked_frame)
    assert len(picked_target) == len(picked_candidate) == len(picked_mirror) == n_picks, (
        f"picked_frame ({n_picks}), picked_target ({len(picked_target)}), "
        f"picked_candidate ({len(picked_candidate)}) and picked_mirror "
        f"({len(picked_mirror)}) must all be the same length."
    )
    assert all(0 <= int(f) < len(frame_stock) for f in picked_frame), (
        f"picked_frame contains an index outside [0, {len(frame_stock) - 1}]."
    )
    assert all(int(m) in (0, 1) for m in picked_mirror), (
        "picked_mirror must contain only 0 or 1."
    )

    # --- State initialization, mirroring BikeBuilder_Env.reset() ---
    stock_mask     = np.ones(len(frame_stock), dtype=np.float32)
    placed_frames  : list[BikeFrame] = []
    connection_log : list[tuple[int, PointDict, PointDict, bool]] = []
    action_log     : list[list[int]] = []
    previous_frame : Optional[BikeFrame] = None
    mirror_flag    : bool = False

    tally = RunTally()

    curve_end         = guide_curve[-1]
    end_tangent       = guide_curve[-1] - guide_curve[-2]
    curve_end_tangent = end_tangent / np.linalg.norm(end_tangent)

    # --- Main loop ---
    for t in range(n_picks):

        # Action unpacking — same order as BikeBuilder_Env.step(): frame, target, candidate, mirror
        frame_idx = int(picked_frame[t])
        target    = PointDict(int(picked_target[t]))       # raises ValueError outside 0-4
        candidate = PointDict(int(picked_candidate[t]))    # raises ValueError outside 0-4
        mirror    = bool(int(picked_mirror[t]))

        # Reuse check — skip, pick consumed (RL: current_step += 1, no placement)
        if stock_mask[frame_idx] == 0.0:
            tally.reuse_skips += 1
            continue

        frame       = frame_stock[frame_idx]
        orientation = (target, candidate, mirror)

        if t == 0:
            # First frame — RL: current_step == 0 branch, no intersection check
            initial_planes = initial_targets(frame, guide_curve)
            placed_frame   = place_candidate(
                frame          = frame,
                orientation    = orientation,
                t              = t,
                mirror_flag    = mirror_flag,
                initial_planes = initial_planes,
            )
        else:
            placed_frame = place_candidate(
                frame          = frame,
                orientation    = orientation,
                t              = t,
                mirror_flag    = mirror_flag,
                previous_frame = previous_frame,
            )

            # Intersection check — skip, pick consumed (RL: current_step += 1, no placement)
            if frames_intersect_proximity(placed_frame, placed_frames, IV.intersect_buffer):
                tally.no_intersect_skips += 1
                continue

        # Commit — RL book keeping
        placed_frames.append(placed_frame)
        connection_log.append((frame_idx, target, candidate, mirror))
        action_log.append([frame_idx, target.value, candidate.value, int(mirror)])
        previous_frame        = placed_frame
        mirror_flag           = mirror
        stock_mask[frame_idx] = 0.0
        tally.steps_completed += 1

        # RL returns before the termination check on the first frame
        if t == 0:
            continue

        # Termination check — current_step is t + 1 after the RL increment
        terminated, _terminal_reward, overshot = check_termination(
            placed_frame, curve_end, curve_end_tangent,
            t + 1, n_picks, strict_termination, full_overshot, terminal_reward_scale
        )

        if terminated:
            tally.terminated = True
            tally.overshot   = overshot
            # Same branching as BikeBuilder_Env.step()
            if full_overshot:
                tally.true_termination = terminated and not overshot
            else:
                tally.true_termination = terminated
            break

    # --- BikeBridge construction, only on true termination ---
    bike_bridge = None
    if tally.true_termination:
        bike_bridge = BikeBridge(placed_frames, connection_log)

    return RunResult(
        placed_frames     = placed_frames,
        connection_log    = connection_log,
        action_log        = action_log,
        remaining_options = [],
        bike_bridge       = bike_bridge,
        tally             = tally,
    )
