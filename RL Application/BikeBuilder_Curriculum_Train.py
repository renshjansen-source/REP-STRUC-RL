# =============================================================================
# IMPORTS  (identical to BikeBuilder_Train.py)
# =============================================================================
import os
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
# DATA IMPORTS (unchanged)
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

seed = seed_everything(1248408448)

# =============================================================================
# CURRICULUM DEFINITION
# =============================================================================
STAGES = [
    {"max_step": 5,  "total_timesteps": 500_000, "note": "Curriculum stage 1/3 - max_step=5"},
    {"max_step": 10, "total_timesteps": 500_000, "note": "Curriculum stage 2/3 - max_step=10"},
    {"max_step": 20, "total_timesteps": 500_000, "note": "Curriculum stage 3/3 - max_step=20"},
]

diary     = TrainingDiary()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir   = f"logs/{timestamp}_curriculum"
os.makedirs(log_dir, exist_ok=True)
print(f"Logging to: {log_dir}")

policy_kwargs = dict(
    features_extractor_class  = PointNet_Extractor,
    features_extractor_kwargs = dict(features_dim=256),
)

model = None

# =============================================================================
# CURRICULUM LOOP
# =============================================================================
for i, stage in enumerate(STAGES, start=1):

    env_kwargs = dict(
        frame_stock   = frame_stock,
        guide_curve   = sampled_curve,
        max_step      = stage["max_step"],
        shuffle_stock = True,
    )

    train_env = make_vec_env(
        "environment/BikeBuilder-v0", n_envs=4, env_kwargs=env_kwargs,
        monitor_dir=f"{log_dir}/stage{i}_train", seed=seed,
    )
    eval_env = make_vec_env(
        "environment/BikeBuilder-v0", n_envs=4, env_kwargs=env_kwargs,
        monitor_dir=f"{log_dir}/stage{i}_eval", seed=seed,
    )

    model_kwargs = dict(
        policy          = "MultiInputPolicy",
        env             = train_env,
        policy_kwargs   = policy_kwargs,
        verbose         = 1,
        tensorboard_log = log_dir,
        device          = "auto",
        seed            = seed,
    )

    if model is None:
        model = PPO(**model_kwargs)          # weights initialized ONCE
    else:
        model.set_env(train_env)             # weights preserved, env swapped

    callback_kwargs = dict(
        eval_env              = eval_env,
        best_model_save_path  = f"{log_dir}/stage{i}_best_model",
        log_path              = f"{log_dir}/stage{i}_eval_logs",
        eval_freq             = 2500,
        n_eval_episodes       = 8,
        deterministic         = True,
        render                = False,
    )
    eval_callback   = EvalCallback(**callback_kwargs)
    custom_callback = BikeBuilder_Callback()

    diary.start(
        run_id          = f"{timestamp}_stage{i}",
        env_class       = BikeBuilder_Env,
        env_kwargs      = env_kwargs,
        model_class     = PPO,
        model_kwargs    = model_kwargs,
        callback_class  = EvalCallback,
        callback_kwargs = callback_kwargs,
        note            = stage["note"],
    )

    print(f"\n=== Stage {i}/{len(STAGES)} — max_step={stage['max_step']} ===")
    model.learn(
        total_timesteps    = stage["total_timesteps"],
        callback            = [eval_callback, custom_callback],
        reset_num_timesteps = (i == 1),   # only stage 1 zeroes the counter
        tb_log_name         = "PPO",      # same name every stage -> one continuous TB run
        progress_bar        = True,
    )

    diary.finish()
    train_env.close()
    eval_env.close()

# =============================================================================
# FINAL SAVE
# =============================================================================
final_model_path = f"{log_dir}/final_model"
model.save(final_model_path)
print(f"Final model saved to: {final_model_path}")