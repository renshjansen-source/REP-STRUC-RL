"""
Export selected TensorBoard scalar tags to one CSV per run.

- First column: time step. No wall time.
- Tags missing from a run are skipped, with a message in the terminal.
- Tags logged at different steps are outer-joined on step (empty cells where a tag has no value).
- A run path may be a log folder (searched recursively for event files) or a single event file.
"""

import os
import glob

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, SCALARS

# =============================================================================
# CONFIG
# =============================================================================

# Run log folder (or a single events.out.tfevents file) -> CSV name (without .csv)
RUNS = {
    "logs/20260930_065829": "entropy_baseline_with_rollout_ep_len",
}

TAGS = [
    "train/explained_variance",
    "train/entropy_loss",
    "termination/true_termination_tally",
    "penalties/intersection_ep_percentage",
    "FEA/frame_compression_mean",
    "FEA/frame_tension_mean",
    "FEA/max_displacement_mean",
    "rollout/ep_len_mean"
]

OUTPUT_DIR = r"C:\Users\rensh\Documents\Master Thesis\Report\Images and Charts\Reinforcement Learning\Results Data\Repo Export Folder"

# =============================================================================
# FUNCTIONS
# =============================================================================

def find_event_files(path):
    if os.path.isfile(path):
        return [path]
    return sorted(glob.glob(os.path.join(path, "**", "events.out.tfevents.*"), recursive=True))


def load_scalars(event_files):
    """Returns {tag: {step: value}} merged over all event files of a run."""
    scalars = {}
    for f in event_files:
        acc = EventAccumulator(f, size_guidance={SCALARS: 0})  # 0 = load all points
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            series = scalars.setdefault(tag, {})
            for event in acc.Scalars(tag):
                series[event.step] = event.value  # later files overwrite duplicate steps
    return scalars


def export_run(run_path, csv_name):
    event_files = find_event_files(run_path)
    if not event_files:
        print(f"[{csv_name}] No event files found in '{run_path}'. Run skipped.")
        return

    scalars = load_scalars(event_files)

    columns = []
    for tag in TAGS:
        if tag not in scalars:
            print(f"[{csv_name}] Tag '{tag}' not found. Skipped.")
            continue
        series = pd.Series(scalars[tag], name=tag)
        columns.append(series)

    if not columns:
        print(f"[{csv_name}] None of the requested tags exist. No CSV written.")
        return

    df = pd.concat(columns, axis=1).sort_index()
    df.index.name = "step"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{csv_name}.csv")
    df.to_csv(out_path)
    print(f"[{csv_name}] Wrote {len(df)} rows, {len(columns)} tag(s) -> {out_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    for run_path, csv_name in RUNS.items():
        export_run(run_path, csv_name)