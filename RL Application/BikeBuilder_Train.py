# =============================================================================
# IMPORTS
# =============================================================================
import os
import gymnasium as gym
import numpy  as np
import pandas as pd
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util  import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

import environment
from internal_variables import IV
from BikeBuilder_Seeder import seed_everything
from environment.envs.BikeBuilder_env_PPO   import BikeBuilder_Env
from environment.envs.BikeBuilder_Utilities import resample_curve, normalized_cross_sections, doubled_tube_section
from environment.envs.BikeBuilder_Classes   import BikeFrame
from BikeBuilder_Custom_Extractor import Custom_PointNet_Extractor
from BikeBuilder_Callback  import BikeBuilder_Callback
from BikeBuilder_Custom_Policy import MaskablePolicy

from Training_Diary import TrainingDiary

# =============================================================================
# DATA IMPORTS
# =============================================================================
# LOADING CURVE
curve_dataframe = pd.read_csv(IV.arch_v0)
guide_curve     = curve_dataframe[["x_cord", "z_cord"]].to_numpy(dtype=np.float32)

shift           = np.array(IV.origin_position, dtype=np.float32) - guide_curve[0]
guide_curve     = (guide_curve + shift).astype(np.float32)

sampled_curve   = resample_curve(guide_curve, IV.curve_samples)

# Loading Bikes - Frames
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

area_maxes = raw_areas.max(axis=0) # Migrate this to normalized_cross_sections
area_maxes[area_maxes == 0] = 1.0
stock_areas = (raw_areas / area_maxes).astype(np.float32)

# =============================================================================
# SEEDING
# =============================================================================
seed = seed_everything(696307358)   # empty = new seed. Current testing seed = 696307358
print(f"Using seed: {seed}")

# =============================================================================
# LOGGING SETUP
# =============================================================================
diary = TrainingDiary()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir   = f"logs/{timestamp}"
os.makedirs(log_dir, exist_ok=True)
print(f"Logging to: {log_dir}")

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
env_kwargs = dict(
    obs_type        = 'mid',       # 'combined' | 'edge' | 'mid' | 'angle'
    stock_mask_mode = 'binary',    # 'binary'   | 'zero_geo' | 'combined_masking' | 'none'
    frame_stock     = frame_stock,
    guide_curve     = sampled_curve,
    stock_areas     = stock_areas,
    max_step        = 25,
    progress_weight = 1.0,
    distance_weight = 1.0,
    use_positive_stock_norm = True,
    shuffle_stock           = True,
    current_frame_sweep     = True,
    enable_termination      = True,
    strict_termination      = False,
    use_stock_areas         = True,
    enable_fea              = True,
    full_overshot           = True,
    normalization_type      = 'bounding', # 'curve' or 'bounding'
)

train_env = make_vec_env(
    "environment/BikeBuilder-v0",
    n_envs      = 16,
    env_kwargs  = env_kwargs,
    monitor_dir = f"{log_dir}/train",
    seed        = seed,
)

eval_env = make_vec_env(
    "environment/BikeBuilder-v0",
    n_envs      = 16,
    env_kwargs  = env_kwargs,
    monitor_dir = f"{log_dir}/eval",
    seed        = seed,
)

# =============================================================================
# KEYWORD ARGUMENTS
# =============================================================================
total_timesteps       = 1_500_000
enable_action_masking = True

policy_kwargs = dict(
    features_extractor_class  = Custom_PointNet_Extractor,
    features_extractor_kwargs = dict(features_dim=256),
    use_masking               = enable_action_masking,
    share_features_extractor  = False,
)

model_kwargs = dict(                          
    policy          = MaskablePolicy,
    env             = train_env,
    policy_kwargs   = policy_kwargs,
    verbose         = 1,
    tensorboard_log = log_dir,
    device          = "auto",
    seed            = seed,
    n_steps         = 256, # 2048 / 4 environments
    batch_size      = 128,
    ent_coef        = 0.0,
    n_epochs        = 10,
)

callback_kwargs = dict(
    eval_env              = eval_env,
    best_model_save_path  = f"{log_dir}/best_model",
    log_path              = f"{log_dir}/eval_logs",
    eval_freq             = 256,
    n_eval_episodes       = 8,
    deterministic         = True,
    render                = False,
)

# =============================================================================
# DIARY START
# =============================================================================
diary.start(
    run_id          = timestamp,
    env_class       = BikeBuilder_Env,
    env_kwargs      = env_kwargs,
    model_class     = PPO,
    model_kwargs    = model_kwargs,
    callback_class  = EvalCallback,
    callback_kwargs = callback_kwargs,
    extra           = dict(
        train_n_envs    = train_env.num_envs,
        eval_n_envs     = eval_env.num_envs,
        total_timesteps = total_timesteps,
    ),
)

# =============================================================================
# TRAINING
# =============================================================================
eval_callback   = EvalCallback(**callback_kwargs) # type: ignore
custom_callback = BikeBuilder_Callback()
model           = PPO(**model_kwargs)             # type: ignore

# --- Parameter count check ---
total_params      = sum(p.numel() for p in model.policy.parameters())
extractor_params  = sum(p.numel() for p in model.policy.features_extractor.parameters())
current_net_params = sum(p.numel() for p in model.policy.features_extractor.current_net.parameters())
print(f"Total policy params:     {total_params}")
print(f"Features extractor params: {extractor_params}")
print(f"current_net params:      {current_net_params}")
# --- end check ---

print("Starting training...")
model.learn(
    total_timesteps = total_timesteps,
    callback        = [eval_callback, custom_callback],
    progress_bar    = True,
)
print("Training complete.")

diary.finish()

# =============================================================================
# SAVING
# =============================================================================
final_model_path = f"{log_dir}/final_model"
model.save(final_model_path)
print(f"Final model saved to: {final_model_path}")

train_env.close()
eval_env.close()