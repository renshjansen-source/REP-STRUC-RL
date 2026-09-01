''' This file contains a dataclass for internal variables used 
in the environment but which aren't specified as keyword arguments '''

from dataclasses import dataclass, field
from pathlib import Path
import math

repo_root = Path(__file__).resolve().parent.parent # REP_STRUC RL/
# print(f"internal_variables.py loaded from: {__file__}")

@dataclass(frozen=True)
class InternalVariables:
    # --- Placement Logics ---
    connection_offset : float = 80.0
    start_rotation    : float = 0.25 * math.pi
    # ---  Data  Handling  ---
    curve_samples     : int   = 100
    curve_norm_range  : int   = 8200
    stock_norm_range  : int   = 630 # Changed from 630 for stock of 25 to 700 for stock of 35 or 50
    # ---  Bounding Space  ---
    origin_position   : tuple[float, float] = (500,500)
    x_bounds          : tuple[float, float] = (0, 8000)
    z_bounds          : tuple[float, float] = (0, 2000)
    # ---  File Locations  ---
    arch_v0           : Path = repo_root / "Datasets" / "Curves" / "final_arch_short_true_0_v0.csv"
    frames_v0         : Path = repo_root / "Datasets" / "Bike Frames" / "FRAMED_new_set_25.csv"
    crs_v0            : Path = repo_root / "Datasets" / "Bike Frames" / "FRAMED_new_set_25_crs.csv"
    # --- Tolerance Variables ---
    intersect_tol        : float = 1e-3
    intersect_buffer     : int   = 3
    termination_vicinity : float = 300.0
    angle_rounding       : int   = 3
    ray_height           : float = 3000.0
    minimum_connection_distance : float = 20.0 # mm
    # --- Behavior  Variables ---
    connection_limit        : float = 200.0
    enable_connection_limit : bool  = True
    enable_timoshenko       : bool  = True
    enable_adaptive_shear   : bool  = True 
    enable_axial_only_stress: bool  = False
    # --- Debugging Variables ---
    FEA_debug : bool = False
    # ---  Reward  Variables  ---
    distance_threshold : float = 500.0
    progress_threshold : float = 1500.0
    termination_step   : float = 3.0
    termination_scale  : tuple[float, float] = (3.0, 1.0) # 3.0 if distance to end point curve = 0
    progress_multiplier: float = 1.2
    progress_exponent  : float = 1.0
    # --- FEA Reward Variables ---
    max_reward_deform      : float = 10.0
    max_reward_tension     : float = 10.0
    max_reward_compression : float = 10.0
    deform_reward          : tuple[float, float] = (2, 50)      # (low, high), cm
    tension_reward         : tuple[float, float] = (70, 2000)   # (low, high), kN/cm²
    compression_reward     : tuple[float, float] = (70, 2000)   # (low, high), kN/cm²
    fea_reward_steepness   : float = 5.0
    # ---  Penalty Variables  ---
    reuse_penalty     : float = -0.5
    ccx_penalty       : float = -2.0
    overshot_penalty  : float = 0.0
    no_progress       : float = 0.0
    # ---   Render Settings   ---
    window_padding    : int   = 100
    grid_colour       : tuple[int, int, int] = (230, 230, 230)
    label_colour      : tuple[int, int, int] = (150, 150, 150)
    g_curve_colour    : tuple[int, int, int] = (200, 200, 200)
    grid_spacing      : float = 1000
    frame_thickness   : int = 3
    rew_label_offset  : float = 800.0
    rew_label_colour  : tuple[int, int, int] = (0, 0, 0)
    leader_colour     : tuple[int, int, int] = (130, 130, 130)
    side_panel_width  : int = 220
    side_panel_colour : tuple[int, int, int] = (245, 245, 245)
    side_panel_text_colour : tuple[int, int, int] = (0, 0, 0)
    side_panel_font_size   : int = 14
    side_panel_line_height : int = 14
    label_font_size   : int = 16
    centroid_colour: tuple[int, int, int] = (255, 0, 0)
    centroid_radius: int = 4
    connector_colour  : tuple[int, int, int] = (0, 150, 150)
    # ---   FEA  pre-flight   ---
    load_divider   : int = 5
    tension_trim   = 10.0         # mm — trim from each end before the intersection check
    tension_count  = 4            # lookahead window when building chords
    tension_thresh = (2, 4000)    # mm — (min, max) allowed chord length
    # ---    FEA Variables    ---
    tributary_width         : float = 1.30                      # in metres
    deck_range              : tuple[float, float] = (0, 8000)   # span of the deck in mm
    connector_OD            : float = 30.0                      # mm
    connector_thickness     : float = 5                         # mm
    tension_OD              : float = 18.0                      # mm
    default_load            : float = 7.50                      # kN/m2
    shear_correction_factor : float = 0.5
    # ---    FEA Materials    ---
    gamma_frames            : float = 78.50                     # kN/m3
    gamma_connection        : float = 78.50                     # kN/m3
    gamma_cable             : float = 81.31                     # kN/m3
    E_frame                 : float = 21_000.0                  # kN/cm2
    E_connection            : float = 21_000.0                  # kN/cm2
    E_cable                 : float = 12_700.0                  # kN/cm2
    G_frames                : float = 8076.0                    # kN/cm2 - inplane and transverse shear modulus
    G_connection            : float = 8076.0                    # kN/cm2 - inplane and transverse shear modulus
    G_cables                : float = 4884.0                    # kN/cm2 - inplane and transverse shear modulus
    # ---    FEA Rendering    ---
    support_radius : int = 6
    pin_colour     : tuple[int, int, int] = (186, 130, 230)
    roller_colour  : tuple[int, int, int] = (98,  0,   150)
    # --- Extractor Behaviour ---
    fuse_mask_in_stock  : bool = True
    fuse_areas_in_stock : bool = False
    # ---  Extractor Layers   ---
    guide_curve_out    = 32
    stock_geometry_out = 64
    stock_areas_out    = 24
    stock_mask_out     = 16
    current_out        = 16
    progress_out       = 8
    

IV  = InternalVariables()