# =============================================================================
# IMPORTS
# =============================================================================
import gymnasium as gym
import numpy as np
import pandas as pd
import pygame

from stable_baselines3 import PPO

import environment
from internal_variables import IV
from environment.envs.BikeBuilder_Utilities import resample_curve
from environment.envs.BikeBuilder_Classes import BikeFrame

# Both extractors need to be importable so PPO.load can resolve whichever one
# the loaded model was actually trained with - it's saved as a class reference,
# not the class body itself.
from BikeBuilder_Extractor import BikeBuilder_Extractor
from PointNet_Extractor import PointNet_Extractor

# =============================================================================
# SETTINGS
# =============================================================================
MODEL_PATH    = "logs/20260821_133113/best_model/best_model"   # <- update to the run you want to visualize
N_EPISODES    = 5
DETERMINISTIC = True

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

# =============================================================================
# ENVIRONMENT
# =============================================================================
env = gym.make(
    "environment/BikeBuilder-v0",
    obs_type        = 'mid',
    stock_mask_mode = 'binary',
    frame_stock        = frame_stock,
    guide_curve        = sampled_curve,
    max_step           = 25,
    progress_weight    = 1.0,
    distance_weight    = 1.0,
    use_positive_stock_norm = True,
    shuffle_stock           = True,
    current_frame_sweep     = True,
    use_stock_areas         = False,
    enable_termination      = True,
    strict_termination      = False,
    visual_debugging    = False,
    normalization_type  = 'bounding',
    render_mode         = 'human',
    render_labels       = True,
)

env.metadata["render_fps"] = 180

# =============================================================================
# MODEL
# =============================================================================
model = PPO.load(MODEL_PATH, env=env)
print(f"Model loaded from: {MODEL_PATH}")

# =============================================================================
# VISUALIZATION LOOP
# =============================================================================
episode_seeds  : list[int] = []
episode_actions: list[list] = []

for episode in range(N_EPISODES):
    ep_seed = int(np.random.randint(0, 2**31 - 1))
    episode_seeds.append(ep_seed)

    obs, info     = env.reset(seed=ep_seed)
    actions_taken : list = []
    episode_reward = 0.0
    print(f"\nEpisode {episode + 1} / {N_EPISODES}  (seed={ep_seed})")

    while True:
        action, _ = model.predict(obs, deterministic=DETERMINISTIC)
        actions_taken.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        episode_reward += reward                                # type: ignore

        if terminated or truncated:
            print(f"  Episode reward: {episode_reward:.4f}")
            break

    episode_actions.append(actions_taken)

# =============================================================================
# KEEP WINDOW OPEN
# =============================================================================
def show_episode(idx: int) -> None:
    env.reset(seed=episode_seeds[idx])
    for action in episode_actions[idx]:
        env.step(action)
    env.render()
    print(f"Showing episode {idx + 1} / {N_EPISODES}")

current_idx = N_EPISODES - 1   # last episode is already on screen from the loop above
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                current_idx = min(current_idx + 1, N_EPISODES - 1)
                show_episode(current_idx)
            elif event.key == pygame.K_LEFT:
                current_idx = max(current_idx - 1, 0)
                show_episode(current_idx)

env.close()