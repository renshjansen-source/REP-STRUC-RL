"""
Batch evaluation of a trained BikeBuilder agent against the FEA allowables in IV.

Runs N_EPISODES, takes the terminal FEA result of each episode, and reports how many
episodes satisfy every allowable. Sign convention: tension positive, compression negative.
Units follow BikeBuilder_FEA output: displacement in cm, stress in kN/cm².
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import pandas as pd

from stable_baselines3 import PPO

import environment
from internal_variables import IV
from environment.envs.BikeBuilder_Utilities import (
    resample_curve,
    doubled_tube_section,
    normalized_cross_sections,
    normalize_stock_areas,
    fea_compliance,
    FEA_CRITERIA,
)
from environment.envs.BikeBuilder_Classes import BikeFrame

# Both extractors need to be importable so PPO.load can resolve whichever one
# the loaded model was actually trained with.
from BikeBuilder_Extractor import BikeBuilder_Extractor
from PointNet_Extractor import PointNet_Extractor

# =============================================================================
# SETTINGS
# =============================================================================
MODEL_PATH    = "logs/20261008_232153_seeding_sweep/453908393_20261009_033910_best_model/best_model"   # <- update to the run you want to evaluate
N_EPISODES    = 1000
DETERMINISTIC = True
BASE_SEED     = 0           # episode k is reset with seed BASE_SEED + k -> reproducible runs
RENDER        = False       # pygame rendering; leave off for long runs
SAVE_CSV      = True        # per-episode results, saved next to the model

# =============================================================================
# DATA LOADING
# =============================================================================
curve_dataframe = pd.read_csv(IV.arch_v0)
guide_curve     = curve_dataframe[["x_cord", "z_cord"]].to_numpy(dtype=np.float32)

shift       = np.array(IV.origin_position, dtype=np.float32) - guide_curve[0]
guide_curve = (guide_curve + shift).astype(np.float32)

sampled_curve = resample_curve(guide_curve, IV.curve_samples)

bikes_dataframe              = pd.read_csv(IV.frames_v0)
frame_stock: list[BikeFrame] = []
for _, row in bikes_dataframe.iterrows():
    points = np.array([
        [row["ST_TOP_X"], row["ST_TOP_Z"]],
        [row["HT_TOP_X"], row["HT_TOP_Z"]],
        [row["HT_BOTTOM_X"], row["HT_BOTTOM_Z"]],
        [row["BB_X"],    row["BB_Z"]],
        [row["CS_SS_X"], row["CS_SS_Z"]],
    ], dtype=np.float32)
    frame_stock.append(BikeFrame(points))

# Loading Bikes - Cross Sectional Areas
crs_dataframe = pd.read_csv(IV.crs_v0)
crs_dataframe = crs_dataframe * 1000.0

# Doubling CS and SS areas
crs_dataframe['CS_OD'], crs_dataframe['CS_T'] = doubled_tube_section(crs_dataframe['CS_OD'], crs_dataframe['CS_T'])
crs_dataframe['SS_OD'], crs_dataframe['SS_T'] = doubled_tube_section(crs_dataframe['SS_OD'], crs_dataframe['SS_T'])

tube_order = ['ST', 'TT', 'HT', 'DT', 'CS', 'SS']
raw_areas = np.array([
    [normalized_cross_sections(float(crs_dataframe.iloc[i][f'{tube}_OD']), float(crs_dataframe.iloc[i][f'{tube}_T']))
     for tube in tube_order]
    for i in range(len(crs_dataframe))
], dtype=np.float32)

stock_areas = normalize_stock_areas(raw_areas)

# =============================================================================
# ENVIRONMENT
# =============================================================================
env = gym.make(
    "environment/BikeBuilder-v0",
    obs_type        = 'mid',       # 'combined' | 'edge' | 'mid' | 'angle'
    stock_mask_mode = 'binary',    # 'binary'   | 'zero_geo' | 'combined_masking' | 'none'
    frame_stock     = frame_stock,
    guide_curve     = sampled_curve,
    stock_areas     = stock_areas,
    max_step        = 35,
    progress_weight = 1.0,
    distance_weight = 1.0,
    use_positive_stock_norm = True,
    shuffle_stock           = True,
    current_frame_sweep     = True,
    current_frame_areas     = False,
    enable_termination      = True,     # required for this script
    strict_termination      = False,
    use_stock_areas         = True,
    enable_fea              = True,     # required for this script
    full_overshot           = True,
    normalization_type      = 'bounding', # 'curve' or 'bounding'
    render_mode             = 'human' if RENDER else None,
    render_labels           = False,
    visual_debugging        = False,
    print_fea_scores        = False,
    reward_debug            = False,
    render_centroids        = False,
    render_vicinity         = False,
)

if RENDER:
    env.metadata["render_fps"] = 180

# =============================================================================
# MODEL
# =============================================================================
model = PPO.load(MODEL_PATH, env=env)
model.set_random_seed(BASE_SEED)            # only matters when DETERMINISTIC = False
print(f"Model loaded from: {MODEL_PATH}")

# =============================================================================
# EVALUATION LOOP
# =============================================================================
STATUS_ORDER = ["no_termination", "overshot", "load_invalid", "tension_invalid", "not_converged", "fea_valid"]

records: list[dict] = []
for episode in range(N_EPISODES):
    ep_seed        = BASE_SEED + episode
    obs, info      = env.reset(seed=ep_seed)
    episode_reward = 0.0

    while True:
        action, _ = model.predict(obs, deterministic=DETERMINISTIC)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += float(reward)                         # type: ignore

        if RENDER:
            env.render()
        if terminated or truncated:
            break

    # Final-step info holds the terminal FEA values (trackers are reset every step)
    if not info["terminated"]:
        status = "no_termination"
    elif not info["true_termination"]:
        status = "overshot"
    elif not info["load_valid"]:
        status = "load_invalid"
    elif not info["tension_valid"]:
        status = "tension_invalid"
    elif not info["fea_valid"]:
        status = "not_converged"
    else:
        status = "fea_valid"

    checks   = fea_compliance(info) if status == "fea_valid" else {}
    all_pass = bool(checks) and all(checks.values())

    record = {
        "episode"        : episode,
        "seed"           : ep_seed,
        "status"         : status,
        "episode_reward" : episode_reward,
        "placed_frames"  : info["placed_frames"],
    }
    for name, (info_key, _, _) in FEA_CRITERIA.items():
        record[info_key]       = info[info_key]
        record[f"pass_{name}"] = checks.get(name)
    record["all_pass"] = all_pass
    records.append(record)

    line = f"[{episode + 1:>5}/{N_EPISODES}] seed={ep_seed:<6} {status:<16} reward={episode_reward:9.3f}"
    if checks:
        failed = [name for name, passed in checks.items() if not passed]
        line  += "  PASS" if all_pass else f"  FAIL: {', '.join(failed)}"
    print(line)

env.close()

# =============================================================================
# REPORT
# =============================================================================
results = pd.DataFrame(records)
fea_df  = results[results["status"] == "fea_valid"]
n_total = len(results)
n_fea   = len(fea_df)

def pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:6.1f}%" if whole else "    n/a"

print("\n" + "=" * 72)
print("FEA COMPLIANCE REPORT")
print("=" * 72)
print(f"Model    : {MODEL_PATH}")
print(f"Episodes : {n_total}   deterministic={DETERMINISTIC}   base_seed={BASE_SEED}")
print("-" * 72)
print("Episode outcomes")
for status in STATUS_ORDER:
    n = int((results["status"] == status).sum())
    print(f"  {status:<34}{n:>6}  {pct(n, n_total)}")
print("-" * 72)
print("Per-criterion pass (of FEA-valid episodes)")
for name, (_, allowable, side) in FEA_CRITERIA.items():
    bound = f"<= {allowable:.2f}" if side == "upper" else f">= {-allowable:.2f}"
    n     = int(fea_df[f"pass_{name}"].astype(bool).sum()) if n_fea else 0
    print(f"  {name:<24}{bound:>10}{n:>6} / {n_fea:<6}{pct(n, n_fea)}")
print("-" * 72)
n_pass = int(fea_df["all_pass"].sum()) if n_fea else 0
print(f"  {'ALL CRITERIA':<34}{n_pass:>6} / {n_fea:<6}{pct(n_pass, n_fea)} of FEA-valid")
print(f"  {'':<34}{'':>15}{pct(n_pass, n_total)} of all episodes")
print("=" * 72)

if SAVE_CSV:
    mode_tag = "det" if DETERMINISTIC else "stoch"
    out_path = os.path.join(
        os.path.dirname(MODEL_PATH),
        f"fea_eval_{n_total}ep_{mode_tag}_{datetime.now():%Y%m%d_%H%M%S}.csv",
    )
    results.to_csv(out_path, index=False)
    print(f"Per-episode results saved to: {out_path}")