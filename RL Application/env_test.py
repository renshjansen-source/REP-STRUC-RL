# =============================================================================
# IMPORTS
# =============================================================================
import os
import gymnasium as gym
import numpy as np
import pandas as pd
import pygame
from datetime import datetime

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

mock_actions = [
    np.array([0, 1, 0, 1]),
    np.array([5, 2, 1, 0]),
    np.array([10, 4, 0, 1]),
    np.array([15, 3, 4, 0]),
    np.array([20, 2, 1, 0]),
    np.array([24, 4, 1, 1]),
    np.array([23, 4, 1, 1]),
    np.array([22, 4, 1, 1]),
    np.array([19, 4, 1, 1]),
    np.array([18, 4, 4, 0]),
    np.array([16, 2, 1, 1]),
    np.array([14, 4, 0, 1]),
    np.array([4, 3, 3, 1]),
]

# =============================================================================
# GYM CREATION
# =============================================================================
env = gym.make(
    "environment/BikeBuilder-v0",
    guide_curve = sampled_curve,
    frame_stock = frame_stock,
    render_mode = "human",
    disable_env_checker= True,
    render_labels = True,
    enable_termination = True,
    strict_termination = True,
    shuffle_stock = False,
    render_centroids = True,
)

env.metadata["render_fps"] = 45

obs, info = env.reset()
for key, space in env.observation_space.spaces.items(): # type: ignore
    if not space.contains(obs[key]):
        print(f"OUT OF BOUNDS: {key}")
        print(f"  actual min/max   : {obs[key].min()} / {obs[key].max()}")
        print(f"  declared low/high: {space.low.min()} / {space.high.max()}")

env.reset()
for action in mock_actions:
    env.step(action)
    env.render()
env.render()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

env.close()

# Action code with proper termination
# mock_actions = [
#     np.array([0, 1, 0, 1]),
#     np.array([5, 2, 1, 0]),
#     np.array([10, 4, 0, 1]),
#     np.array([15, 3, 4, 0]),
#     np.array([20, 2, 1, 0]),
#     np.array([24, 4, 1, 1]),
#     np.array([23, 4, 1, 1]),
#     np.array([22, 4, 1, 1]),
#     np.array([19, 4, 1, 1]),
#     np.array([18, 4, 4, 0]),
#     np.array([16, 2, 1, 1]),
#     np.array([14, 4, 0, 1]),
#     np.array([4, 3, 3, 1]),
# ]

# Action code with strict termination failure
# mock_actions = [
#     np.array([0, 1, 0, 1]),
#     np.array([5, 2, 1, 0]),
#     np.array([10, 4, 0, 1]),
#     np.array([15, 3, 4, 0]),
#     np.array([20, 2, 1, 0]),
#     np.array([24, 4, 1, 1]),
#     np.array([23, 4, 1, 1]),
#     np.array([22, 4, 1, 1]),
#     np.array([19, 4, 1, 1]),
#     np.array([18, 4, 4, 0]),
#     np.array([16, 2, 1, 1]),
#     np.array([14, 4, 1, 0]),
#     np.array([1, 3, 1, 1]),
# ]