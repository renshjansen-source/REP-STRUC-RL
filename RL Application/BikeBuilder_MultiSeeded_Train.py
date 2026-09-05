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
from environment.envs.BikeBuilder_Utilities import resample_curve, doubled_tube_section, normalized_cross_sections
from environment.envs.BikeBuilder_Classes   import BikeFrame
from BikeBuilder_Callback  import BikeBuilder_Callback
from Training_Diary import TrainingDiary
from BikeBuilder_Custom_Extractor import Custom_PointNet_Extractor
from BikeBuilder_Custom_Policy import MaskablePolicy

# =============================================================================
# SWEEP DEFINITION
# =============================================================================
CONFIGS = [
    {"seed": 230923904,  "note": "Seeding sweep with seed 230923904"},
    {"seed": 453908393,  "note": "Seeding sweep with seed 453908393"},
    {"seed": 123948430,  "note": "Seeding sweep with seed 123948430"},
    {"seed": 112233303,  "note": "Seeding sweep with seed 112233303"},
    {"seed": 432940201,  "note": "Seeding sweep with seed 432940201"},
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
# LOGGING SETUP — one parent folder for the whole sweep
# =============================================================================
diary          = TrainingDiary()
sweep_stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
sweep_log_dir  = f"logs/{sweep_stamp}_seeding_sweep"
os.makedirs(sweep_log_dir, exist_ok=True)
failures_path  = f"{sweep_log_dir}/failures.txt"
print(f"Sweep logging to: {sweep_log_dir}")

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
total_timesteps       = 2_000_000
enable_action_masking = True

policy_kwargs = dict(
    features_extractor_class  = Custom_PointNet_Extractor,
    features_extractor_kwargs = dict(features_dim=256),
    use_masking               = enable_action_masking,
    share_features_extractor  = False,
)

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

# =============================================================================
# SWEEP LOOP
# =============================================================================
for i, cfg in enumerate(CONFIGS, start=1):

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"{cfg['seed']}_{run_stamp}"
    print(f"\n=== Run {i}/{len(CONFIGS)} — {run_name} ===")

    # -------------------------------------------------------------------------
    # SEEDING
    # -------------------------------------------------------------------------

    seed = seed_everything(cfg["seed"])
    print(f"Using seed: {seed}")

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
        model_kwargs = dict(
            policy          = MaskablePolicy,
            env             = train_env,
            policy_kwargs   = policy_kwargs,
            n_steps         = 256,
            batch_size      = 128,
            verbose         = 1,
            tensorboard_log = sweep_log_dir,
            device          = "auto",
            seed            = seed,
            target_kl       = 0.06,
        )

        callback_kwargs = dict(
            eval_env               = eval_env,
            best_model_save_path   = f"{sweep_log_dir}/{run_name}_best_model",
            log_path               = f"{sweep_log_dir}/{run_name}_eval_logs",
            eval_freq              = 256,         # One per rollout
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
            extra           = dict(
                train_n_envs    = train_env.num_envs,
                eval_n_envs     = eval_env.num_envs,
                total_timesteps = total_timesteps,
            ),
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