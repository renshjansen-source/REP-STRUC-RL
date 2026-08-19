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
from environment.envs.BikeBuilder_Utilities import coordinate_to_pixel, initial_targets, place_first, place, frames_intersect, step_reward, check_termination, build_observation_points, build_current_frame_observation
from environment.envs.BikeBuilder_Classes import PointDict, BikeFrame, ShapeGrammar, EpisodeGrammar

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
            frame_stock  : list[BikeFrame],             # BikeFrame Objects
            guide_curve  : np.ndarray,                  # Pre-sampled in the training file
            stock_areas  = None,
            max_step     = 25,
            window_scale = 5,
            render_mode  = None,
            distance_weight = 1.0,
            progress_weight = 1.0,
            normalization_type  = 'curve',              # 'curve' or 'bounding'
            current_frame_sweep = False,
            shuffle_stock       = True,                     # Added: Now stock can be shuffled
            use_stock_mask      = False,                    # Added: Allows for switching between stock zeroing and stock masking for reuse
            use_stock_areas     = False,                    # Added: Optional stock area injector
            render_labels       = False,                    # Added: Allows for more extensive rendering
            render_centroids    = False,
            enable_termination : bool = False,          # Added: Enables termination logics
            strict_termination : bool = False,          # Added: If true, termination only yields rewards if the frame has not exceeded the curve
    ):
        # Datasets
        self.guide_curve = guide_curve
        self.frame_stock = frame_stock

        # Observation Variables
        self.use_stock_mask      = use_stock_mask
        self.use_stock_areas     = use_stock_areas
        self.stock_areas         = stock_areas              # Stock areas are pre-normalized in training script
        self.obs_type            = obs_type
        self.max_step            = max_step
        self.current_frame_sweep = current_frame_sweep
        self.buffer_size         = IV.intersect_buffer if self.current_frame_sweep else None
        self.normalization_type  = normalization_type

        # Determine obs shapes
        self.points_per_frame = build_observation_points(
            self.frame_stock[0], self.obs_type, IV.stock_norm_range
        ).shape[0]

        self.current_frame_shape = (
            (IV.intersect_buffer, self.points_per_frame, 2) if self.current_frame_sweep
            else (self.points_per_frame, 2)
        )

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
                low   = -1.0,                                   # Changed - Used to be 0.0 but the BB normalization can cause negative values
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
        if self.use_stock_mask:
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
        self.terminated: bool   = False
        self.enable_termination = enable_termination
        self.strict_termination = strict_termination

        self.curve_end  = self.guide_curve[-1]
        end_tangent     = self.guide_curve[-1] - self.guide_curve[-2]
        self.curve_end_tangent = end_tangent / np.linalg.norm(end_tangent)

        # Render Initialization
        self.render_labels    = render_labels
        self.render_centroids = render_centroids
        self.window_scale  = window_scale
        self.canvas_size   = [x_max // self.window_scale, z_max // self.window_scale]
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
        
        if self.use_stock_mask:
            obs["stock_mask"] = self.stock_mask
        if self.use_stock_areas:
            obs["stock_areas"] = self.stock_areas_episode
        
        return obs

    def _get_info(self):
        return {
        "placed_frames": len(self.placed_frames),
        "current_step" : self.current_step,
        "max_t"        : self.max_t,
        "ccx_count"    : self.ccx_counter,
        "reuse_count"  : self.reuse_counter,
        "d_reward"     : self.d_reward,
        "p_reward"     : self.p_reward,
        "terminated"   : self.terminated,
        }
    # ─────────────────────────────────────────────────────────────────────────
    # ACTION MASKING FUNCTION
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

        # Initializing stock permutation
        if self.shuffle_stock:
            self.perm = self.np_random.permutation(len(self.frame_stock))
        else:
            self.perm = np.arange(len(self.frame_stock))

        self.stock_geometry_episode  = self.stock_geometry_norm[self.perm]

        if self.use_stock_areas:
            self.stock_areas_episode = self.stock_areas[self.perm]              # type: ignore

        # Initializing counters
        self.current_step  = 0
        self.reuse_counter = False
        self.ccx_counter   = False
        self.terminated    = False
        self.grammar.reset()

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
        terminated      = False
        terminal_reward = 0.0
        self.terminated = False

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
            if not self.use_stock_mask:
                self.stock_geometry_episode[action[0]]  = 0.0       # Changed: Stock is now zeroed, no stock mask
                if self.use_stock_areas:
                    self.stock_areas_episode[action[0]] = 0.0
            self.current_step  += 1
            
            truncated = self.current_step >= self.max_step
            obs       = self._get_obs()
            info      = self._get_info()
            return obs, reward, terminated, truncated, info

        # Default Frame Placement
        assert self.previous_frame is not None
        placed_frame = place(frame, self.previous_frame, candidate, target, mirror, self.mirror_flag)        

        # Intersection Check
        buffer_frames = self.placed_frames[-IV.intersect_buffer:] # Only last few frames
        if frames_intersect(placed_frame, buffer_frames):
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
        if not self.use_stock_mask:
            self.stock_geometry_episode[action[0]]  = 0.0
            if self.use_stock_areas:
                self.stock_areas_episode[action[0]] = 0.0
        self.current_step += 1

        # Termination Check
        if self.enable_termination:
            terminated, terminal_reward = check_termination(
                placed_frame, self.curve_end, self.curve_end_tangent,
                self.current_step, self.max_step, self.strict_termination
            )
            reward += terminal_reward

        self.placed_frames.append(placed_frame)
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

        # Rendering Grid
        x = 0
        while x <= self.bounds["x_max"]:
            px = coordinate_to_pixel((x, 0), self.canvas_size, self.bounds, self.bounding_range)[0]
            pygame.draw.line(canvas, IV.grid_colour, (px, 0), (px, self.canvas_size[1]), 1)
            label = self.font.render(f"{int(x / 1000)}m", True, IV.label_colour)
            canvas.blit(label, (px - label.get_width() // 2, self.canvas_size[1] - label.get_height() - 2))
            x += IV.grid_spacing
        
        z = 0
        while z <= self.bounds["z_max"]:
            pz = coordinate_to_pixel((0, z), self.canvas_size, self.bounds, self.bounding_range)[1]
            pygame.draw.line(canvas, IV.grid_colour, (0, pz), (self.canvas_size[0], pz), 1)
            label = self.font.render(f"{int(z / 1000)}m", True, IV.label_colour)
            canvas.blit(label, (2, pz - label.get_height() // 2))
            z += IV.grid_spacing
        
        # Drawing the guide curve
        curve_pixels = [coordinate_to_pixel(p, self.canvas_size, self.bounds, self.bounding_range) for p in self.guide_curve]
        pygame.draw.lines(canvas, IV.g_curve_colour, False, curve_pixels, 2)

        # Drawing centroids
        if self.render_centroids:
            for frame in self.placed_frames:
                centroid_px = coordinate_to_pixel(frame.Centroid, self.canvas_size, self.bounds, self.bounding_range)
                pygame.draw.circle(canvas, IV.centroid_colour, centroid_px, IV.centroid_radius)

        # Drawing labels for frames
        if self.render_labels:
            for i, (frame, (d_r, p_r, t_r)) in enumerate(zip(self.placed_frames, self.placement_rewards)):
                points = frame.points
                j      = [coordinate_to_pixel(p, self.canvas_size, self.bounds, self.bounding_range) for p in points]
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

                anchor_px = coordinate_to_pixel(anchor_world, self.canvas_size, self.bounds, self.bounding_range)
                label_px  = coordinate_to_pixel(label_world,  self.canvas_size, self.bounds, self.bounding_range)

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
                j      = [coordinate_to_pixel(p, self.canvas_size, self.bounds, self.bounding_range) for p in points]
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

    def close(self) :
            if self.window is not None:
                pygame.display.quit()
                pygame.quit()