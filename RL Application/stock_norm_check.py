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

overall_max = raw_areas.max()
if overall_max == 0:
    overall_max = 1.0
stock_areas = (raw_areas / overall_max).astype(np.float32)
print(overall_max)
print(stock_areas)