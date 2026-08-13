# =============================================================================
# IMPORTS
# =============================================================================
import os
import traceback
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
from environment.envs.BikeBuilder_Utilities import resample_curve
from environment.envs.BikeBuilder_Classes   import BikeFrame
from PointNet_Extractor    import PointNet_Extractor
from BikeBuilder_Callback  import BikeBuilder_Callback
from Training_Diary import TrainingDiary

# =============================================================================
# SWEEP DEFINITION
# =============================================================================
CONFIGS = [
    {"n_steps": 128, "batch_size": 64,  "note": "HP sweep: n_steps=128, batch_size=64, n_envs=16"},
    {"n_steps": 256, "batch_size": 64,  "note": "HP sweep: n_steps=256, batch_size=64, n_envs=16"},
    {"n_steps": 256, "batch_size": 128, "note": "HP sweep: n_steps=256, batch_size=128, n_envs=16"},
    {"n_steps": 512, "batch_size": 64,  "note": "HP sweep: n_steps=512, batch_size=64, n_envs=16"},
    {"n_steps": 512, "batch_size": 128, "note": "HP sweep: n_steps=512, batch_size=128, n_envs=16"},
    {"n_steps": 512, "batch_size": 256, "note": "HP sweep: n_steps=512, batch_size=256, n_envs=16"},
]

# =============================================================================
# DATA IMPORTS
# =============================================================================
curve_dataframe = pd.read_csv(IV.arch_v0)
guide_curve     = curve_dataframe[["x_cord", "z_cord"]].to_numpy(dtype=np.float32)
shift           = np.array(IV.origin_position, dtype=np.float32) - guide_curve[0]
guide_curve     = (guide_curve + shift).astype(np.float32)
sampled_curve   = resample_curve(guide_curve, IV.curve_samples)

bikes_dataframe = pd.read_csv(IV.frames_v0)
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
# LOGGING SETUP — one parent folder for the whole sweep
# =============================================================================
diary          = TrainingDiary()
sweep_stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
sweep_log_dir  = f"logs/{sweep_stamp}_hp_sweep"
os.makedirs(sweep_log_dir, exist_ok=True)
failures_path  = f"{sweep_log_dir}/failures.txt"
print(f"Sweep logging to: {sweep_log_dir}")

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
policy_kwargs = dict(
    features_extractor_class  = PointNet_Extractor,
    features_extractor_kwargs = dict(features_dim=256),
)

env_kwargs = dict(
    frame_stock   = frame_stock,
    guide_curve   = sampled_curve,
    max_step      = 10,
    shuffle_stock = True,
)

# =============================================================================
# SWEEP LOOP
# =============================================================================
for i, cfg in enumerate(CONFIGS, start=1):

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"{cfg['n_steps']}_{cfg['batch_size']}_{run_stamp}"
    print(f"\n=== Run {i}/{len(CONFIGS)} — {run_name} ===")

    # -------------------------------------------------------------------------
    # ENVIRONMENT SETUP
    # -------------------------------------------------------------------------
    try:
        train_env = make_vec_env(
            "environment/BikeBuilder-v0",
            n_envs      = 16,
            env_kwargs  = env_kwargs,
            monitor_dir = f"{sweep_log_dir}/{run_name}_train",
            seed        = seed,
        )

        eval_env = make_vec_env(
            "environment/BikeBuilder-v0",
            n_envs      = 16,
            env_kwargs  = env_kwargs,
            monitor_dir = f"{sweep_log_dir}/{run_name}_eval",
            seed        = seed,
        )

        # ---------------------------------------------------------------------
        # KEYWORD ARGUMENTS
        # ---------------------------------------------------------------------   
        total_timesteps = 500_000

        model_kwargs = dict(
            policy          = "MultiInputPolicy",
            env             = train_env,
            policy_kwargs   = policy_kwargs,
            n_steps         = cfg["n_steps"],
            batch_size      = cfg["batch_size"],
            verbose         = 1,
            tensorboard_log = sweep_log_dir,
            device          = "auto",
            seed            = seed,
        )

        callback_kwargs = dict(
            eval_env              = eval_env,
            best_model_save_path  = f"{sweep_log_dir}/{run_name}_best_model",
            log_path              = f"{sweep_log_dir}/{run_name}_eval_logs",
            eval_freq              = cfg["n_steps"],         # One per rollout
            n_eval_episodes        = 8,
            deterministic          = True,
            render                 = False,
        )
        # ---------------------------------------------------------------------
        # DIARY SETUP
        # ---------------------------------------------------------------------
        diary.start(
            run_id          = run_name,
            env_class       = BikeBuilder_Env,
            env_kwargs      = env_kwargs,
            model_class     = PPO,
            model_kwargs    = model_kwargs,
            callback_class  = EvalCallback,
            callback_kwargs = callback_kwargs,
            note            = cfg["note"],
            train_n_envs    = train_env.num_envs,
            eval_n_envs     = eval_env.num_envs,
            total_timesteps = total_timesteps,
        )

        # ---------------------------------------------------------------------
        # TRAINING
        # ---------------------------------------------------------------------

        model = PPO(**model_kwargs)                         # type: ignore
        eval_callback   = EvalCallback(**callback_kwargs)   # type: ignore
        custom_callback = BikeBuilder_Callback()

        model.learn(
            total_timesteps  = total_timesteps,
            callback         = [eval_callback, custom_callback],
            tb_log_name      = run_name,
            progress_bar     = True,
        )

        # ---------------------------------------------------------------------
        # SAVING
        # ---------------------------------------------------------------------
        diary.finish()
        model.save(f"{sweep_log_dir}/{run_name}_final_model")

        train_env.close()
        eval_env.close()

    # -------------------------------------------------------------------------
    # EXCEPTIONS
    # -------------------------------------------------------------------------
    except Exception as e:
        print(f"!!! Run {run_name} failed: {e}")
        with open(failures_path, "a", encoding="utf-8") as f:
            f.write(f"{run_name}\n{traceback.format_exc()}\n{'-'*60}\n")
        continue

print("\nSweep complete.")