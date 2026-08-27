# =============================================================================
# IMPORTS
# =============================================================================
import gymnasium as gym
import numpy as np
import pandas as pd
import pygame

import environment
from internal_variables import IV
from environment.envs.BikeBuilder_Utilities import resample_curve
from environment.envs.BikeBuilder_Classes import BikeFrame

# =============================================================================
# DATA IMPORTS
# =============================================================================

# Loading Curve
curve_dataframe = pd.read_csv(IV.arch_v0)
guide_curve     = curve_dataframe[["x_cord", "z_cord"]].to_numpy(dtype=np.float32)

shift           = np.array(IV.origin_position, dtype=np.float32) - guide_curve[0]
guide_curve     = (guide_curve + shift).astype(np.float32)

sampled_curve   = resample_curve(guide_curve, IV.curve_samples)

# Loading Bikes
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

# =============================================================================
# MOCK ACTIONS
# =============================================================================
# Action order: [frame_idx, attach_tar, attach_cand, mirror]

mock_actions = [
    np.array([5,4,1,0]),
    np.array([6,3,1,0]),
    np.array([15,4,2,1]),
    np.array([3,0,1,1]),
    np.array([2,3,2,1]),
    np.array([19,0,2,0]),
    np.array([16,0,0,1]),
    np.array([7,3,4,1]),
    np.array([9,1,3,1]),
    np.array([23,0,3,1]),
    np.array([0,0,2,1]),
    np.array([1,0,0,0]),
    np.array([22,3,2,0]),
]

# =============================================================================
# GYM CREATION
# =============================================================================
env = gym.make(
    "environment/BikeBuilder-v0",
    obs_type            = 'mid',      # 'combined' | 'edge' | 'mid' | 'angle'
    stock_mask_mode      = 'binary',  # 'binary' | 'zero_geo' | 'combined_masking' | 'none'
    frame_stock          = frame_stock,
    guide_curve          = sampled_curve,
    render_mode          = "human",
    disable_env_checker  = True,
    shuffle_stock        = False,
    use_stock_areas      = False,
    render_labels        = True,
    render_centroids     = True,
    enable_termination   = True,
    strict_termination   = False,
    visual_debugging     = True,
)

env.metadata["render_fps"] = 45

# =============================================================================
# OBSERVATION BOUNDS CHECK
# =============================================================================
obs, info = env.reset()
for key, space in env.observation_space.spaces.items():  # type: ignore
    if key not in obs:
        print(f"MISSING FROM OBS: {key}")
        continue
    if not space.contains(obs[key]):
        print(f"OUT OF BOUNDS: {key}")
        print(f"  actual min/max   : {obs[key].min()} / {obs[key].max()}")
        print(f"  declared low/high: {space.low.min()} / {space.high.max()}")

# =============================================================================
# MOCK ACTION ROLLOUT
# =============================================================================
env.reset()
for action in mock_actions:
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    print(
        f"reward={reward:.4f} | terminated={terminated} | truncated={truncated} "
        f"| ccx={info['ccx_count']} | reuse={info['reuse_count']} "
        f"| true_term={info['true_termination']}"
    )
    if terminated or truncated:
        break
env.render()

# =============================================================================
# KEEP WINDOW OPEN
# =============================================================================
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

env.close()