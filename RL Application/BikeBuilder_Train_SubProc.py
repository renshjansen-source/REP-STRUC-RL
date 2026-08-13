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
from stable_baselines3.common.vec_env   import SubprocVecEnv

import environment
from internal_variables import IV
from BikeBuilder_Seeder import seed_everything
from environment.envs.BikeBuilder_env_PPO   import BikeBuilder_Env
from environment.envs.BikeBuilder_Utilities import resample_curve
from environment.envs.BikeBuilder_Classes   import BikeFrame
from BikeBuilder_Extractor import BikeBuilder_Extractor
from PointNet_Extractor    import PointNet_Extractor
from BikeBuilder_Callback  import BikeBuilder_Callback

from Training_Diary import TrainingDiary

def main():
    # =============================================================================
    # DATA IMPORTS
    # =============================================================================

    # LOADING CURVE
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
    # SEEDING
    # =============================================================================

    seed = seed_everything(696307358)   # empty = new seed
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
        frame_stock   = frame_stock,
        guide_curve   = sampled_curve,
        max_step      = 25,
        shuffle_stock = True,
        use_stock_mask     = True,
        enable_termination = True,
        strict_termination = False,
    )

    train_env = make_vec_env(
        "environment/BikeBuilder-v0",
        n_envs      = 8,
        env_kwargs  = env_kwargs,
        monitor_dir = f"{log_dir}/train",
        seed        = seed,
        vec_env_cls = SubprocVecEnv,
    )

    eval_env = make_vec_env(
        "environment/BikeBuilder-v0",
        n_envs      = 8,
        env_kwargs  = env_kwargs,
        monitor_dir = f"{log_dir}/eval",
        seed        = seed,
        vec_env_cls = SubprocVecEnv,
    )

    # =============================================================================
    # KEYWORD ARGUMENTS
    # =============================================================================
    policy_kwargs = dict(
        features_extractor_class  = PointNet_Extractor,
        features_extractor_kwargs = dict(features_dim=256),
    )

    model_kwargs = dict(                          
        policy          = "MultiInputPolicy",
        env             = train_env,
        policy_kwargs   = policy_kwargs,
        verbose         = 1,
        tensorboard_log = log_dir,
        device          = "auto",
        seed            = seed,
        n_steps         = 256, # 2048 / 4 environments
        batch_size      = 64,
    )

    callback_kwargs = dict(
        eval_env              = eval_env,
        best_model_save_path  = f"{log_dir}/best_model",
        log_path              = f"{log_dir}/eval_logs",
        eval_freq             = 2500,
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
    )

    # =============================================================================
    # TRAINING
    # =============================================================================
    eval_callback   = EvalCallback(**callback_kwargs) # type: ignore
    custom_callback = BikeBuilder_Callback()
    model           = PPO(**model_kwargs)             # type: ignore

    print("Starting training...")
    model.learn(
        total_timesteps = 500_000,
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

if __name__ == "__main__":
    main()