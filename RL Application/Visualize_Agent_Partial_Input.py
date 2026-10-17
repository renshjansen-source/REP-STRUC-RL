"""
Grid capture of agent completions from a prescribed partial state, rendered in stress mode.

Each episode first replays PREFIX_ACTIONS (the partial structure), then the deployed agent
completes it. Every episode is captured, successful or not.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os

import gymnasium as gym
import numpy as np
import pandas as pd
import pygame

from stable_baselines3 import PPO

import environment
from internal_variables import IV
from environment.envs.BikeBuilder_Utilities import resample_curve, doubled_tube_section, normalized_cross_sections, normalize_stock_areas
from environment.envs.BikeBuilder_Classes import BikeFrame

# Both extractors need to be importable so PPO.load can resolve whichever one
# the loaded model was actually trained with - it's saved as a class reference,
# not the class body itself.
from BikeBuilder_Extractor import BikeBuilder_Extractor
from PointNet_Extractor import PointNet_Extractor

# =============================================================================
# SETTINGS
# =============================================================================
MODEL_PATH    = "logs/20260930_065829/best_model/best_model"   # <- update to the run you want to visualize
DETERMINISTIC = True
GRID_COLUMNS  = 3
GRID_ROWS     = 4
IMAGE_NAME    = "grid_prefix_3_4_1_test"
OUTPUT_PATH   = rf"C:\Users\rensh\Documents\Master Thesis\Report\Images and Charts\Reinforcement Learning\Results Data\Repo Export Grid Images\{IMAGE_NAME}.png"

# Partial state: [raw_frame_idx, attach_tar, attach_cand, mirror]
# Same format as env_test mock_actions — frame index refers to the CSV row, NOT the shuffled slot.
PREFIX_ACTIONS = [
    np.array([19,3,3,0]),
    np.array([7,1,2,1]),
    np.array([23,4,3,1]),
    np.array([1,0,3,1]),
    np.array([16,1,3,0]),
    np.array([4,1,3,0]),
    # np.array([20,0,2,0]),
    # np.array([15,4,0,1]),
]

    # [19, 3, 3, 0],
    # [ 7, 1, 2, 1],
    # [23, 4, 3, 1],
    # [ 1, 0, 3, 1],
    # [16, 1, 3, 0],
    # [ 4, 1, 3, 0],

# =============================================================================
# EXPORT GUARD
# =============================================================================
if os.path.exists(OUTPUT_PATH):
    raise FileExistsError(f"Output image already exists, set a new IMAGE_NAME: {OUTPUT_PATH}")

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
    enable_termination      = True,
    strict_termination      = False,
    use_stock_areas         = True,
    enable_fea              = True,
    full_overshot           = True,
    print_fea_scores = True,
    normalization_type      = 'bounding', # 'curve' or 'bounding'
    render_mode             = 'stress',
)

env.metadata["render_fps"] = 180

# =============================================================================
# MODEL
# =============================================================================
model = PPO.load(MODEL_PATH, env=env)
print(f"Model loaded from: {MODEL_PATH}")

# =============================================================================
# GRID CAPTURE
# =============================================================================
n_episodes   = GRID_COLUMNS * GRID_ROWS
grid_surface = None

for episode in range(n_episodes):
    ep_seed   = int(np.random.randint(0, 2**31 - 1))
    obs, info = env.reset(seed=ep_seed)

    # Replay the partial state. Raw frame index -> this episode's shuffled slot.
    for raw_idx, target, candidate, mirror in PREFIX_ACTIONS:
        slot = int(np.flatnonzero(env.unwrapped.perm == raw_idx)[0])
        n_placed_before = info["placed_frames"]
        obs, reward, terminated, truncated, info = env.step(np.array([slot, target, candidate, mirror]))

        if info["placed_frames"] != n_placed_before + 1:
            raise RuntimeError(f"Prefix action {[raw_idx, target, candidate, mirror]} was not placed (reuse or intersection).")
        if terminated or truncated:
            raise RuntimeError(f"Prefix action {[raw_idx, target, candidate, mirror]} ended the episode — prefix is not a partial state.")

    # Agent completes the structure
    while True:
        action, _ = model.predict(obs, deterministic=DETERMINISTIC)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # Render the final state only, then copy it into its grid cell (row-major)
    env.render()
    frame_surface = pygame.display.get_surface()
    tile_width, tile_height = frame_surface.get_size()

    if grid_surface is None:
        grid_surface = pygame.Surface((GRID_COLUMNS * tile_width, GRID_ROWS * tile_height))

    column = episode % GRID_COLUMNS
    row    = episode // GRID_COLUMNS
    grid_surface.blit(frame_surface, (column * tile_width, row * tile_height))

    print(f"Episode {episode + 1} / {n_episodes}  (seed={ep_seed})")

assert grid_surface is not None
pygame.image.save(grid_surface, OUTPUT_PATH)
print(f"Saved {GRID_COLUMNS}x{GRID_ROWS} grid to: {OUTPUT_PATH}")

env.close()