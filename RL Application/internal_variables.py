''' This file contains a dataclass for internal variables used 
in the environment but which aren't specified as keyword arguments '''

from dataclasses import dataclass, field
from pathlib import Path
import math

repo_root = Path(__file__).resolve().parent.parent # REP_STRUC RL/

@dataclass(frozen=True)
class InternalVariables:
    # --- Placement Logics ---
    connection_offset : float = 50.0
    start_rotation    : float = 0.25 * math.pi
    # ---  Data  Handling  ---
    curve_samples     : int   = 100
    curve_norm_range  : int   = 8200
    stock_norm_range  : int   = 630
    # ---  Bounding Space  ---
    origin_position   : tuple[float, float] = (1000,2000)
    x_bounds          : tuple[float, float] = (0, 9000)
    z_bounds          : tuple[float, float] = (0, 4500)
    # ---  File Locations  ---
    arch_v0           : Path = repo_root / "Datasets" / "Curves" / "final_arch_short_v0.csv"
    frames_v0         : Path = repo_root / "Datasets" / "Bike Frames" / "FRAMED_new_set_25.csv"
    crs_v0            : Path = repo_root / "Datasets" / "Bike Frames" / "FRAMED_new_set_25_crs.csv"
    # --- Tolerance Variables ---
    intersect_tol     : float = 1e-9
    intersect_buffer  : int   = 3
    termination_vicinity : float = 300.0
    # ---  Reward  Variables  ---
    distance_threshold: float = 500.0
    progress_threshold: float = 1500.0
    no_progress       : float = 0.0
    termination_step  : float = 1.2
    # ---  Penalty Variables  ---
    reuse_penalty     : float = 0.0
    ccx_penalty       : float = 0.0
    overshot_penalty  : float = 0.0
    # --- Render  Settings ---
    window_padding    : float = 0.20
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
    side_panel_font_size   : int = 16
    side_panel_line_height : int = 18
    label_font_size   : int = 16
    centroid_colour: tuple[int, int, int] = (255, 0, 0)
    centroid_radius: int = 4
    

IV  = InternalVariables()