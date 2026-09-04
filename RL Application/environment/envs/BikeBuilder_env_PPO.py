# =============================================================================
# IMPORTS
# =============================================================================

import gymnasium as gym
import pygame
import math
import numpy as np
import pandas as pd

from gymnasium import spaces

from internal_variables import IV
from environment.envs.BikeBuilder_Utilities import (
    coordinate_to_pixel, 
    initial_targets, 
    place_first, 
    place, 
    frames_intersect, 
    step_reward, 
    check_termination, 
    build_observation_points, 
    build_current_frame_observation,
    build_observation_points_positive,
    frames_intersect_proximity,
    fea_reward,
    FEA_reward_recip,
    )

from environment.envs.BikeBuilder_Classes import PointDict, BikeFrame, ShapeGrammar, EpisodeGrammar, BikeBridge
from environment.envs.BikeBuilder_FEA import run_fea, print_fea_result

# =============================================================================
# Environment Class
# =============================================================================

class BikeBuilder_Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "sleek"], "render_fps": 4}
    # ─────────────────────────────────────────────────────────────────────────
    # INITIALIZATION
    # ─────────────────────────────────────────────────────────────────────────
    def __init__(
            self,
            obs_type,                                   # 'combined' | 'edge' | 'mid' | 'angle'
            stock_mask_mode,                            # 'binary'   | 'zero_geo' | 'combined_masking' | 'none'
            frame_stock  : list[BikeFrame],             # BikeFrame Objects
            guide_curve  : np.ndarray,                  # Pre-sampled in the training file
            stock_areas  = None,
            max_step     = 25,
            window_scale = 8,
            render_mode  = None,
            distance_weight = 1.0,
            progress_weight = 1.0,
            normalization_type  = 'curve',              # 'curve' or 'bounding'
            use_positive_stock_norm   = False,
            current_frame_sweep       = False,
            shuffle_stock             = True,                     # Added: Now stock can be shuffled
            use_stock_areas           = False,                    # Added: Optional stock area injector
            enable_fea         : bool = False,
            render_labels             = False,                    # Added: Allows for more extensive rendering
            render_centroids          = False,
            visual_debugging          = False,
            enable_termination : bool    = False,          # Added: Enables termination logics
            strict_termination : bool    = False,          # Added: If true, termination only yields rewards if the frame has not exceeded the curve
            full_overshot      : bool    = True,           # Added: Can be used to bypass the final overshot check. 
            terminal_reward_scale : bool = False,
    ):
        # Datasets
        self.guide_curve = guide_curve
        self.frame_stock = frame_stock

        # Enable FEA
        self.enable_fea  = enable_fea

        # Observation Variables
        self.use_stock_areas         = use_stock_areas
        self.stock_areas             = stock_areas              # Stock areas are pre-normalized in training script
        self.obs_type                = obs_type
        self.max_step                = max_step
        self.current_frame_sweep     = current_frame_sweep
        self.buffer_size             = self.max_step if self.current_frame_sweep else None
        self.normalization_type      = normalization_type
        self.use_positive_stock_norm = use_positive_stock_norm

        # Stock masking Variables
        valid_modes = ("binary", "zero_geo", "combined_masking", "none")
        if stock_mask_mode not in valid_modes:
            raise ValueError(
                f"Unknown stock_mask_mode {stock_mask_mode!r} - expected one of {valid_modes}"
            )
        self.stock_mask_mode    = stock_mask_mode
        self.use_stock_mask_obs = stock_mask_mode in ("binary", "combined_masking")
        self.zero_on_consume    = stock_mask_mode in ("zero_geo", "combined_masking")

        # Determine obs shapes
        self.points_per_frame = build_observation_points(
            self.frame_stock[0], self.obs_type, IV.stock_norm_range
        ).shape[0]

        if self.current_frame_sweep:
            assert self.buffer_size is not None, (
                "buffer_size must be set when current_frame_sweep is True."
            )
            self.current_frame_shape = (self.buffer_size, self.points_per_frame, 2)
        else:
            self.current_frame_shape = (self.points_per_frame, 2)
            
        if self.obs_type == "angle":
            stock_geom_low = -1.0
        elif self.use_positive_stock_norm:
            stock_geom_low = 0.0
        else:
            stock_geom_low = -1.0 

        # Bounding Area Variables
        x_min, x_max = IV.x_bounds
        z_min, z_max = IV.z_bounds

        # Bounding Area Dictionary
        self.bounds = {
            "x_min": x_min,
            "x_max": x_max,
            "z_min": z_min,
            "z_max": z_max,
        }
        self.bounding_range = [x_max - x_min, z_max - z_min]

        # Normalization Check
        if self.normalization_type   == 'bounding':
            self._norm_xy = np.array([x_max, z_max], dtype=np.float32)
        elif self.normalization_type == 'curve':
            self._norm_xy = np.array([
            self.guide_curve[:, 0].max(),
            self.guide_curve[:, 1].max(),
            ], dtype=np.float32)
        else:
            raise ValueError(
                f"Unknown normalization_type {self.normalization_type!r} — expected 'bounding' or 'curve'"
            )

        # Action Space
        self.action_space = spaces.MultiDiscrete([len(self.frame_stock), 5, 5, 2])

        # Observation Space
        obs_dict: dict[str, spaces.Space] = {
            "guide_curve": spaces.Box(
                low   = 0.0,
                high  = 1.0,                                    # Changed - Normalized, used to be IV.guide_norm_range
                shape = (IV.curve_samples, 2),
                dtype = np.float32
            ),
            "stock_geometry": spaces.Box(
                low   = stock_geom_low,                         # Changed - Used to be 0.0 but the BB normalization can cause negative values
                high  = 1.0,                                    # Changed - Normalized, used to be IV.stock_norm_range               
                shape = (len(self.frame_stock), self.points_per_frame, 2),
                dtype = np.float32
            ),                                                  # Changed - stock mask has been removed
            "current_frame": spaces.Box(                        # Changed - now contains both edge and mid points
                low   = -np.inf,
                high  = np.inf,
                shape = self.current_frame_shape,               # Changed - allows for sweeping observation
                dtype = np.float32
            ),
            "progress": spaces.Box(
                low = 0.0,
                high = 1.0,
                shape = (1,),
                dtype = np.float32
            ),
            "max_t": spaces.Box(
                low   = 0.0,
                high  = 1.0,
                shape = (1,),
                dtype = np.float32
            ),
        }

        # Toggleable Observations
        if self.use_stock_mask_obs:
            obs_dict["stock_mask"] = spaces.Box(
                low=0.0, high=1.0, shape=(len(self.frame_stock),), dtype=np.float32
            )  

        if self.use_stock_areas:
            obs_dict["stock_areas"] = spaces.Box(
                low=0.0, high=1.0, shape=(len(self.frame_stock), 6), dtype=np.float32
            )

        self.observation_space = spaces.Dict(obs_dict)

        # Normalization
        self.guide_curve_norm    = (self.guide_curve / self._norm_xy).astype(np.float32)
        if self.use_positive_stock_norm:
            self.stock_geometry_norm = build_observation_points_positive(self.frame_stock, self.obs_type)
        else:
            self.stock_geometry_norm = np.array([
                build_observation_points(frame, self.obs_type, IV.stock_norm_range)
                for frame in self.frame_stock
            ], dtype=np.float32)


        # Randomization Variables
        self.shuffle_stock  = shuffle_stock

        # Tracking Variables
        self.grammar  = EpisodeGrammar(
            stock     = ShapeGrammar(size=len(self.frame_stock)),
            target    = ShapeGrammar(size=len(PointDict), labels=[p.name for p in PointDict]),
            candidate = ShapeGrammar(size=len(PointDict), labels=[p.name for p in PointDict]),
            mirror    = ShapeGrammar(size=2, labels=["original", "mirrored"]),
        )

        # Reward initialization
        self.distance_weight = distance_weight
        self.progress_weight = progress_weight

        # Termination initialization
        self.terminated :       bool = False
        self.overshot :         bool = False
        self.true_termination : bool = False
        self.enable_termination = enable_termination
        self.strict_termination = strict_termination
        self.full_overshot      = full_overshot
        self.terminal_reward_scale = terminal_reward_scale

        self.curve_end  = self.guide_curve[-1]
        end_tangent     = self.guide_curve[-1] - self.guide_curve[-2]
        self.curve_end_tangent = end_tangent / np.linalg.norm(end_tangent)

        # Render Initialization
        self.render_labels    = render_labels
        self.render_centroids = render_centroids
        self.window_scale     = window_scale
        self.visual_debugging = visual_debugging
        # Unpadded drawing size — drives world-to-pixel scale, unchanged from before
        self.draw_size = [x_max // window_scale, z_max // window_scale]
        # Padded canvas — actual pygame Surface size for the drawing area
        self.canvas_size = [self.draw_size[0] + 2 * IV.window_padding,
        self.draw_size[1] + 2 * IV.window_padding]
        self.window_size   = [self.canvas_size[0] + IV.side_panel_width, self.canvas_size[1]]
        self.window        = None
        self.clock         = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVATION FUNCTION
    # ─────────────────────────────────────────────────────────────────────────
    def _get_obs(self):

        current_frame = build_current_frame_observation(
            self.placed_frames, self.obs_type, self._norm_xy, self.points_per_frame, self.buffer_size
        )

        obs = {
        "guide_curve"    : self.guide_curve_norm,
        "stock_geometry" : self.stock_geometry_episode,                                                 # Changed: Stock mask has been removed
        "current_frame"  : current_frame,
        "progress"       : np.array([self.current_step / self.max_step], dtype=np.float32),
        "max_t"          : np.array([self.max_t], dtype=np.float32),
        }
        
        if self.use_stock_mask_obs:
            obs["stock_mask"] = self.stock_mask
        if self.use_stock_areas:
            obs["stock_areas"] = self.stock_areas_episode
        
        return obs

    def _get_info(self):
        return {
        "placed_frames"    : len(self.placed_frames),
        "current_step"     : self.current_step,
        "max_t"            : self.max_t,
        "ccx_count"        : self.ccx_counter,
        "reuse_count"      : self.reuse_counter,
        "d_reward"         : self.d_reward,
        "p_reward"         : self.p_reward,
        "terminated"       : self.terminated,
        "overshot"         : self.overshot,
        "true_termination" : self.true_termination,
        "load_valid"       : self.load_valid,
        "tension_valid"    : self.tension_valid,
        "deform_r"         : self.deform_r,
        "tension_r"        : self.tension_r,
        "compression_r"    : self.compression_r,
        "fea_max_disp"     : self.fea_max_disp,
        "frame_sig_max"    : self.frame_sig_max,
        "frame_sig_min"    : self.frame_sig_min,
        "connector_sig_max": self.connector_sig_max,
        "connector_sig_min": self.connector_sig_min,
        "cable_sig_max"    : self.cable_sig_max,
        "cable_sig_min"    : self.cable_sig_min,
        "fea_ran"          : self.fea_ran,
        "fea_valid"        : self.fea_valid,
        }
    # ─────────────────────────────────────────────────────────────────────────
    # ACTION MASKING FUNCTION (outdated - i think)
    # ─────────────────────────────────────────────────────────────────────────
    def action_masks(self) -> np.ndarray:
        return np.concatenate([
            self.stock_mask.astype(bool),
            np.ones(len(PointDict), dtype=bool),
            np.ones(len(PointDict), dtype=bool),
            np.ones(2, dtype=bool),
        ])
    
    # ─────────────────────────────────────────────────────────────────────────
    # RESET FUNCTION
    # ─────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):                    # type: ignore
        super().reset(seed=seed)

        # Initializing variables
        self.max_t         : np.float32 = np.float32(0.0)
        self.placed_frames : list[BikeFrame]  = []
        self.previous_frame: BikeFrame | None = None
        self.mirror_flag   : bool             = False
        self.stock_mask    : np.ndarray       = np.ones(len(self.frame_stock), dtype=np.float32)

        # FEA initialization
        self.bike_bridge : BikeBridge | None = None
        self.fea_result  : dict | None       = None

        # Initializing stock permutation
        if self.shuffle_stock:
            self.perm = self.np_random.permutation(len(self.frame_stock))
        else:
            self.perm = np.arange(len(self.frame_stock))

        self.stock_geometry_episode  = self.stock_geometry_norm[self.perm]

        if self.use_stock_areas:
            self.stock_areas_episode = self.stock_areas[self.perm]              # type: ignore

        # Initializing counters
        self.current_step     = 0
        self.reuse_counter    = False
        self.ccx_counter      = False
        self.terminated       = False
        self.overshot         = False
        self.true_termination = False
        self.load_valid       = False
        self.tension_valid    = False
        self.fea_valid        = False
        self.grammar.reset()

        # FEA Trackers
        self.fea_ran          = False
        self.load_valid       = False
        self.tension_valid    = False

        self.deform_r      = 0.0
        self.tension_r     = 0.0
        self.compression_r = 0.0

        self.fea_max_disp        = None
        self.frame_sig_max       = None
        self.frame_sig_min       = None
        self.connector_sig_max   = None
        self.connector_sig_min   = None
        self.cable_sig_max       = None
        self.cable_sig_min       = None

        # Connection log for BikeBridge class
        self.connection_log: list[tuple[int, PointDict, PointDict, bool]] = []

        # Initializing sub-rewards
        self.p_reward      = 0
        self.d_reward      = 0

        # Initializing trackers for rendering
        if self.render_labels:
            self.placement_rewards: list[tuple[float, float, float]] = []
            self.action_log       : list[str]                 = []

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    # ─────────────────────────────────────────────────────────────────────────
    # STEP FUNCTION
    # ─────────────────────────────────────────────────────────────────────────  
    def step(self, action):                                      # type: ignore

        # Setting exit conditions
        terminated            = False
        terminal_reward       = 0.0
        self.terminated       = False
        self.overshot         = False
        self.true_termination = False
        self.load_valid       = False
        self.tension_valid    = False

        # Resetting FEA trackers
        self.fea_ran          = False
        self.load_valid       = False
        self.tension_valid    = False
        self.fea_valid        = False
        self.deform_r      = 0.0
        self.tension_r     = 0.0
        self.compression_r = 0.0
        self.fea_max_disp        = None
        self.frame_sig_max       = None
        self.frame_sig_min       = None
        self.connector_sig_max   = None
        self.connector_sig_min   = None
        self.cable_sig_max       = None
        self.cable_sig_min       = None

        # Setting reuse and ccx flags
        self.reuse_counter = False
        self.ccx_counter   = False

        # Action 0 | Stock selection with unshuffling
        slot_idx  = action[0]
        raw_idx   = self.perm[slot_idx]
        frame     = self.frame_stock[raw_idx]

        # Action 1 | Target selection
        target    = PointDict(action[1])

        # Action 2 | Candidate selection
        candidate = PointDict(action[2])

        # Action 3 | Mirror Boolean
        mirror    = bool(action[3])

        # Action Recording
        self.grammar.record(action)

        # Action Recording for rendering
        if self.render_labels:
            action_code = f"{raw_idx}-{target.name[:2]}-{candidate.name[:2]}-{int(action[3])}"

        # Reuse Check
        if self.stock_mask[action[0]] == 0.0:
            self.current_step  += 1
            reward              = IV.reuse_penalty
            truncated           = self.current_step >= self.max_step
            self.reuse_counter  = True
            if self.render_labels:
                self.action_log.append(action_code + "-RU") # type: ignore
            
            obs                 = self._get_obs()
            info                = self._get_info()
            return obs, reward, terminated, truncated, info

        # First Frame Placement
        if self.current_step == 0:
            initial_planes = initial_targets(frame, self.guide_curve)
            initial_frame  = place_first(frame, initial_planes, candidate, target, mirror)
            # Fetch Rewards
            reward, self.max_t, self.d_reward, self.p_reward = step_reward(
                initial_frame, self.guide_curve, self.max_t, 
                self.progress_weight, self.distance_weight
            )
            self.placed_frames.append(initial_frame)
            if self.render_labels:
                self.placement_rewards.append((float(self.d_reward), float(self.p_reward), 0.0))
                self.action_log.append(action_code)                 # type: ignore
            self.previous_frame = initial_frame
            self.mirror_flag    = mirror
            self.stock_mask[action[0]] = 0.0
            if self.zero_on_consume:
                self.stock_geometry_episode[action[0]]  = 0.0       # Changed: Stock is now zeroed, no stock mask
                if self.use_stock_areas:
                    self.stock_areas_episode[action[0]] = 0.0
            self.connection_log.append((int(raw_idx), target, candidate, mirror))
            self.current_step  += 1
            
            truncated = self.current_step >= self.max_step
            obs       = self._get_obs()
            info      = self._get_info()
            return obs, reward, terminated, truncated, info

        # Default Frame Placement
        assert self.previous_frame is not None
        placed_frame = place(frame, self.previous_frame, candidate, target, mirror, self.mirror_flag)        

        # Intersection Check
        # buffer_frames = self.placed_frames[-IV.intersect_buffer:] # Only last few frames
        # if frames_intersect(placed_frame, buffer_frames):
        if frames_intersect_proximity(placed_frame, self.placed_frames, IV.intersect_buffer):
            self.current_step += 1
            self.ccx_counter   = True
            reward    = IV.ccx_penalty
            truncated = self.current_step >= self.max_step        # CLEAR PLACED FRAME?
            if self.render_labels:
                self.action_log.append(action_code + "-CCX")     # type: ignore

            obs       = self._get_obs()
            info      = self._get_info()
            return obs, reward, terminated, truncated, info

        # Fetch Rewards
        reward, self.max_t, self.d_reward, self.p_reward = step_reward(
            placed_frame, self.guide_curve, self.max_t, 
            self.progress_weight, self.distance_weight
        )

        # Book Keeping
        self.previous_frame = placed_frame
        self.mirror_flag    = mirror 
        self.stock_mask[action[0]] = 0.0   
        if self.zero_on_consume:
            self.stock_geometry_episode[action[0]]  = 0.0
            if self.use_stock_areas:
                self.stock_areas_episode[action[0]] = 0.0
        self.connection_log.append((int(raw_idx), target, candidate, mirror))
        self.current_step += 1

        # Termination Check
        if self.enable_termination:
            terminated, terminal_reward, overshot = check_termination(
                placed_frame, self.curve_end, self.curve_end_tangent,
                self.current_step, self.max_step, self.strict_termination,
                self.full_overshot, self.terminal_reward_scale
            )
            reward        += terminal_reward
            if self.full_overshot:
                self.true_termination = terminated == True and overshot == False
            else:
                self.true_termination = terminated == True
            self.overshot  = overshot

        self.placed_frames.append(placed_frame)

        # FINITE ELEMENT ANALYSIS
        if self.true_termination:
            self.bike_bridge = BikeBridge(self.placed_frames, self.connection_log)
            self.load_valid      = self.bike_bridge.load_valid
            self.tension_valid   = self.bike_bridge.tension_valid

            if self.enable_fea and self.load_valid and self.tension_valid:
                self.fea_result = run_fea(self.bike_bridge)
                self.fea_ran    = True
                # print_fea_result(self.fea_result)
                if self.fea_result["converged"] == True:
                    self.deform_r, self.tension_r, self.compression_r = fea_reward(self.fea_result)
                    fea_total = self.deform_r + self.tension_r + self.compression_r
                    reward += fea_total
                    self.fea_valid = True

                    # Trackers for logging
                    self.fea_max_disp      = self.fea_result["max_displacement"]
                    self.frame_sig_max     = self.fea_result["frame_stress"]["sig_max"]
                    self.frame_sig_min     = self.fea_result["frame_stress"]["sig_min"]
                    self.connector_sig_max = self.fea_result["connector_stress"]["sig_max"]
                    self.connector_sig_min = self.fea_result["connector_stress"]["sig_min"]
                    self.cable_sig_max     = self.fea_result["cable_stress"]["sig_max"]
                    self.cable_sig_min     = self.fea_result["cable_stress"]["sig_min"]

                else:
                    self.fea_valid = False
            else:
                self.fea_result = None

        if self.render_labels:
            self.placement_rewards.append((float(self.d_reward), float(self.p_reward), float(terminal_reward)))
            if terminated:
                self.action_log.append(action_code + "-TERM")   # type: ignore
            else:
                self.action_log.append(action_code)             # type: ignore

        self.terminated = terminated
        truncated       = self.current_step >= self.max_step
        
        obs = self._get_obs()
        info      = self._get_info()
        return obs, reward, terminated, truncated, info
             
    # ─────────────────────────────────────────────────────────────────────────
    # RENDER DISPATCH
    # ─────────────────────────────────────────────────────────────────────────
    
    def render(self):
        if self.render_mode == "human":
            return self.render_canvas()
    
    def render_canvas(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.window = pygame.display.set_mode(self.window_size)
            self.font   = pygame.font.SysFont('Arial', 10)
            self.label_font = pygame.font.SysFont('Arial', IV.label_font_size)
            self.panel_font = pygame.font.SysFont('Arial', IV.side_panel_font_size)
        if self.clock is None:                                          # Technically this check is redundant
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface(self.window_size)
        canvas.fill((255,255,255)) # White

        if self.visual_debugging and self.bike_bridge is not None:
            self.render_debug(canvas)
        else:

            # Rendering Grid
            x = 0
            while x <= self.bounds["x_max"]:
                px = coordinate_to_pixel((x, 0), self.draw_size, self.bounds, self.bounding_range)[0]
                pygame.draw.line(canvas, IV.grid_colour, (px, 0), (px, self.canvas_size[1]), 1)
                label = self.font.render(f"{int(x / 1000)}m", True, IV.label_colour)
                canvas.blit(label, (px - label.get_width() // 2, self.canvas_size[1] - label.get_height() - 2))
                x += IV.grid_spacing
            
            z = 0
            while z <= self.bounds["z_max"]:
                pz = coordinate_to_pixel((0, z), self.draw_size, self.bounds, self.bounding_range)[1]
                pygame.draw.line(canvas, IV.grid_colour, (0, pz), (self.canvas_size[0], pz), 1)
                label = self.font.render(f"{int(z / 1000)}m", True, IV.label_colour)
                canvas.blit(label, (2, pz - label.get_height() // 2))
                z += IV.grid_spacing
            
            # Drawing the guide curve
            curve_pixels = [coordinate_to_pixel(p, self.draw_size, self.bounds, self.bounding_range) for p in self.guide_curve]
            pygame.draw.lines(canvas, IV.g_curve_colour, False, curve_pixels, 2)

            # Drawing centroids
            if self.render_centroids:
                for frame in self.placed_frames:
                    centroid_px = coordinate_to_pixel(frame.Centroid, self.draw_size, self.bounds, self.bounding_range)
                    pygame.draw.circle(canvas, IV.centroid_colour, centroid_px, IV.centroid_radius)

            # Drawing labels for frames
            if self.render_labels:
                for i, (frame, (d_r, p_r, t_r)) in enumerate(zip(self.placed_frames, self.placement_rewards)):
                    points = frame.points
                    j      = [coordinate_to_pixel(p, self.draw_size, self.bounds, self.bounding_range) for p in points]
                    t      = IV.frame_thickness
                    pygame.draw.line(canvas, (0, 0, 0), j[0], j[1], t)  # TT
                    pygame.draw.line(canvas, (0, 0, 0), j[1], j[2], t)  # HT
                    pygame.draw.line(canvas, (0, 0, 0), j[2], j[3], t)  # DT
                    pygame.draw.line(canvas, (0, 0, 0), j[3], j[4], t)  # CS
                    pygame.draw.line(canvas, (0, 0, 0), j[4], j[0], t)  # SS
                    pygame.draw.line(canvas, (0, 0, 0), j[3], j[0], t)  # ST

                    # Reward label — offset from centroid, alternating up/down by placement order
                    direction    = 1 if i % 2 == 0 else -1
                    anchor_world = frame.Centroid
                    label_world  = anchor_world + np.array([0.0, direction * IV.rew_label_offset], dtype=np.float32)

                    anchor_px = coordinate_to_pixel(anchor_world, self.draw_size, self.bounds, self.bounding_range)
                    label_px  = coordinate_to_pixel(label_world,  self.draw_size, self.bounds, self.bounding_range)

                    margin   = 60
                    label_px = (
                        int(np.clip(label_px[0], margin, self.canvas_size[0] - margin)),
                        int(np.clip(label_px[1], 20, self.canvas_size[1] - 20)),
                    )

                    pygame.draw.line(canvas, IV.leader_colour, anchor_px, label_px, 1)

                    label_text = f"d:{d_r:.2f} p:{p_r:.2f}"
                    if t_r != 0.0:
                        label_text += f" t:{t_r:.2f}"
                    label_surface = self.label_font.render(label_text, True, IV.rew_label_colour)
                    canvas.blit(label_surface, (label_px[0] - label_surface.get_width() // 2,
                                                label_px[1] - label_surface.get_height() // 2))
            else:
                # Draw the bike frames
                for frame in self.placed_frames:
                    points = frame.points
                    j      = [coordinate_to_pixel(p, self.draw_size, self.bounds, self.bounding_range) for p in points]
                    t      = IV.frame_thickness

                    pygame.draw.line(canvas, (0, 0, 0), j[0], j[1], t)  # TT
                    pygame.draw.line(canvas, (0, 0, 0), j[1], j[2], t)  # HT
                    pygame.draw.line(canvas, (0, 0, 0), j[2], j[3], t)  # DT
                    pygame.draw.line(canvas, (0, 0, 0), j[3], j[4], t)  # CS
                    pygame.draw.line(canvas, (0, 0, 0), j[4], j[0], t)  # SS
                    pygame.draw.line(canvas, (0, 0, 0), j[3], j[0], t)  # ST

            # Draw the side panel
            if self.render_labels:
                panel_x = self.canvas_size[0]
                pygame.draw.rect(canvas, IV.side_panel_colour, (panel_x, 0, IV.side_panel_width, self.window_size[1]))

                panel_y = 10
                for entry in self.action_log:
                    entry_surface = self.panel_font.render(entry, True, IV.side_panel_text_colour)
                    canvas.blit(entry_surface, (panel_x + 10, panel_y))
                    panel_y += IV.side_panel_line_height
                    if panel_y > self.window_size[1] - IV.side_panel_line_height:
                        break   # stop rather than overflow past the visible window

        # Build Canvas
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def render_debug(self, canvas):
        dot_radius       = 4
        dot_colour       = (0, 0, 0)
        label_colour     = (255, 0, 0)
        label_font_size  = 12
        label_radius     = 18      # pixel distance the label sits out from its point, along the centroid→point direction
        leader_colour    = (150, 150, 150)
        leader_thickness = 1
        tube_thickness   = 3
        connection_colour    = (0, 128, 128)
        connection_thickness = 2
        pin_colour        = (186, 130, 230)
        roller_colour     = (98, 0, 150)
        support_radius    = 6
        load_point_colour = (255, 0, 0)
        load_point_radius = 7
        tension_colour   = (218, 165, 32)
        tension_thickness = 2

        debug_print_frame = 8   # set to None to disable, or an int frame index to inspect

        tube_colours = {
            "top_tube"   : (198, 244, 178),
            "head_tube"  : (154, 224, 130),
            "down_tube"  : (105, 197, 90),
            "chain_stay" : (63, 161, 60),
            "seat_stay"  : (34, 120, 40),
            "seat_tube"  : (10, 74, 24),
        }

        debug_font = pygame.font.SysFont('Arial', label_font_size)
        bridge     = self.bike_bridge
        assert bridge is not None

        if debug_print_frame is not None and debug_print_frame < len(bridge.points):
            print(f"\n--- Frame {debug_print_frame} point order ---")
            for local_idx, point in enumerate(bridge.points[debug_print_frame]):
                print(f"  [{local_idx}] {point}")
            print(f"Corner index map: {bridge.corner_index[debug_print_frame]}")

        tube_sets = {
            "top_tube"   : bridge.top_tubes,
            "head_tube"  : bridge.head_tubes,
            "down_tube"  : bridge.down_tubes,
            "chain_stay" : bridge.chain_stays,
            "seat_stay"  : bridge.seat_stays,
            "seat_tube"  : bridge.seat_tubes,
        }

        # Draw tubes first, so points/labels sit visibly on top
        for tube_name, per_frame_indices in tube_sets.items():
            colour = tube_colours[tube_name]
            for frame_idx, index_list in enumerate(per_frame_indices):
                assert len(index_list) >= 2, (
                    f"{tube_name} on frame {frame_idx} resolved to fewer than 2 points — "
                    f"check BikeBridge.build_points()."
                )
                frame_points = bridge.points[frame_idx]
                pixel_points = [
                    coordinate_to_pixel(frame_points[idx], self.draw_size, self.bounds, self.bounding_range)
                    for idx in index_list
                ]
                pygame.draw.lines(canvas, colour, False, pixel_points, tube_thickness)
                
        # Draw every point as a dot, labelled with its local storage order,
        # offset radially outward from the frame's centroid, with a leader line back to it
        for frame_idx, frame in enumerate(bridge.placed_frames):
            frame_points = bridge.points[frame_idx]
            centroid_px  = coordinate_to_pixel(frame.Centroid, self.draw_size, self.bounds, self.bounding_range)

            for local_idx, point in enumerate(frame_points):
                px = coordinate_to_pixel(point, self.draw_size, self.bounds, self.bounding_range)
                pygame.draw.circle(canvas, dot_colour, px, dot_radius)

                direction = np.array([centroid_px[0] - px[0], centroid_px[1] - px[1]], dtype=np.float32)
                norm      = np.linalg.norm(direction)
                assert norm > 1e-6, (
                    f"Point {local_idx} on frame {frame_idx} coincides with its own centroid — "
                    f"cannot compute a label direction."
                )
                unit_dir = direction / norm

                label_px = (
                    int(px[0] + unit_dir[0] * label_radius),
                    int(px[1] + unit_dir[1] * label_radius),
                )

                pygame.draw.line(canvas, leader_colour, px, label_px, leader_thickness)

                label_surface = debug_font.render(str(local_idx), True, label_colour)
                canvas.blit(label_surface, (
                    label_px[0] - label_surface.get_width()  // 2,
                    label_px[1] - label_surface.get_height() // 2,
                ))

        # Draw connections
        for connection_set in bridge.connections:
            for triple in connection_set:
                pixel_points = [
                    coordinate_to_pixel(bridge.points[frame_idx][local_idx], self.draw_size, self.bounds, self.bounding_range)
                    for frame_idx, local_idx in triple
                ]
                pygame.draw.lines(canvas, connection_colour, False, pixel_points, connection_thickness)

        # Draw supports
        pin_frame_idx, pin_local_idx = bridge.pin
        pin_point = bridge.points[pin_frame_idx][pin_local_idx]
        pin_px    = coordinate_to_pixel(pin_point, self.draw_size, self.bounds, self.bounding_range)
        pygame.draw.circle(canvas, pin_colour, pin_px, support_radius)

        roller_frame_idx, roller_local_idx = bridge.roller
        roller_point = bridge.points[roller_frame_idx][roller_local_idx]
        roller_px    = coordinate_to_pixel(roller_point, self.draw_size, self.bounds, self.bounding_range)
        pygame.draw.circle(canvas, roller_colour, roller_px, support_radius)

        # Draw tension lines
        for (start_frame, start_local), (end_frame, end_local) in bridge.tension_lines:
            start_point = bridge.points[start_frame][start_local]
            end_point   = bridge.points[end_frame][end_local]
            start_px = coordinate_to_pixel(start_point, self.draw_size, self.bounds, self.bounding_range)
            end_px   = coordinate_to_pixel(end_point,   self.draw_size, self.bounds, self.bounding_range)
            pygame.draw.line(canvas, tension_colour, start_px, end_px, tension_thickness)

        # Draw load points (red) — list is empty if load_valid is False, so nothing renders in that case
        for frame_idx, local_idx in bridge.load_points:
            point    = bridge.points[frame_idx][local_idx]
            point_px = coordinate_to_pixel(point, self.draw_size, self.bounds, self.bounding_range)
            pygame.draw.circle(canvas, load_point_colour, point_px, load_point_radius)

    def close(self) :
            if self.window is not None:
                pygame.display.quit()
                pygame.quit()