"""
FEA calculation for the terminal evaluation.
Built against BikeBridge — node identity is ID-based (frame_idx, local_idx),
not coordinate-based. No manual element splitting: BikeBridge.build_points
already spliced connector points into each tube's index sequence.
"""
# pyright: reportAttributeAccessIssue=false
# =============================================================================
# IMPORTS
# =============================================================================
import math
import pandas as pd
import openseespy.opensees as ops

from internal_variables import IV
from environment.envs.BikeBuilder_Utilities import doubled_tube_section
from environment.envs.BikeBuilder_Classes   import BikeBridge

# =============================================================================
# DATA HANDLING
# =============================================================================
_crs_dataframe = pd.read_csv(r"C:\Users\rensh\Documents\Master Thesis\REP_STRUC RL\Datasets\Bike Frames\FRAMED_new_set_25_crs.csv") * 1000.0
_crs_dataframe['CS_OD'], _crs_dataframe['CS_T'] = doubled_tube_section(_crs_dataframe['CS_OD'], _crs_dataframe['CS_T'])
_crs_dataframe['SS_OD'], _crs_dataframe['SS_T'] = doubled_tube_section(_crs_dataframe['SS_OD'], _crs_dataframe['SS_T'])

# =============================================================================
# SECTION PROPERTIES
# =============================================================================
def pipe_section(outer_diameter: float, thickness: float) -> tuple[float, float]:
    # Returns (A, Iz) for a hollow circular tube, in mm² and mm⁴
    inner_diameter = outer_diameter - 2.0 * thickness
    A  = (math.pi / 4.0)  * (outer_diameter**2 - inner_diameter**2)
    Iz = (math.pi / 64.0) * (outer_diameter**4 - inner_diameter**4)
    return A, Iz

def solid_circle_area(diameter: float) -> float:
    # Returns A for a solid circular cross-section, in mm²
    return (math.pi / 4.0) * diameter**2

# =============================================================================
# NODE REGISTRY
# =============================================================================
def register_node(id_to_tag: dict, id_to_xz: dict, bridge: BikeBridge, point_id: tuple[int, int]) -> int:
    if point_id in id_to_tag:
        return id_to_tag[point_id]

    tag = len(id_to_tag) + 1   # OpenSeesPy node tags are 1-indexed
    id_to_tag[point_id] = tag

    frame_idx, local_idx = point_id
    x, z = bridge.points[frame_idx][local_idx]
    id_to_xz[point_id] = (float(x), float(z))

    return tag

# Tube Name Mapping — BikeBridge property name -> CRS column prefix
TUBE_PROPERTIES = {
    "top_tubes"   : "TT",
    "head_tubes"  : "HT",
    "down_tubes"  : "DT",
    "chain_stays" : "CS",
    "seat_stays"  : "SS",
    "seat_tubes"  : "ST",
}

# =============================================================================
# ELEMENT CONSTRUCTION — FRAME TUBES
# =============================================================================
def build_frame_elements(bridge, id_to_tag: dict, id_to_xz: dict, elements: list) -> None:
    for property_name, tube_prefix in TUBE_PROPERTIES.items():
        per_frame_indices = getattr(bridge, property_name)

        for frame_idx, index_list in enumerate(per_frame_indices):
            stock_idx = bridge.stock_indices[frame_idx]
            row       = _crs_dataframe.iloc[stock_idx]

            outer_diameter = float(row[f"{tube_prefix}_OD"])
            thickness      = float(row[f"{tube_prefix}_T"])
            A, Iz = pipe_section(outer_diameter, thickness)
            W = Iz / (outer_diameter / 2.0)

            for k in range(len(index_list) - 1):
                point_a_id = (frame_idx, index_list[k])
                point_b_id = (frame_idx, index_list[k + 1])

                node_a = register_node(id_to_tag, id_to_xz, bridge, point_a_id)
                node_b = register_node(id_to_tag, id_to_xz, bridge, point_b_id)

                elements.append(("beam", "frame", node_a, node_b, A, Iz, W))

# =============================================================================
# ELEMENT CONSTRUCTION — CONNECTORS
# =============================================================================
def build_connector_elements(bridge: BikeBridge, id_to_tag: dict, id_to_xz: dict, elements: list) -> None:
    A, Iz = pipe_section(IV.connector_OD, IV.connector_thickness)
    W = Iz / (IV.connector_OD / 2.0)

    for connection_set in bridge.connections:
        for entry in connection_set:
            for k in range(len(entry) - 1):
                point_a_id = entry[k]
                point_b_id = entry[k + 1]

                node_a = register_node(id_to_tag, id_to_xz, bridge, point_a_id)
                node_b = register_node(id_to_tag, id_to_xz, bridge, point_b_id)

                elements.append(("beam", "connector", node_a, node_b, A, Iz, W))

# =============================================================================
# ELEMENT CONSTRUCTION — CABLES
# =============================================================================
def build_cable_elements(bridge: BikeBridge, id_to_tag: dict, id_to_xz: dict, elements: list) -> None:
    A = solid_circle_area(IV.tension_OD)

    for point_a_id, point_b_id in bridge.tension_lines:
        node_a = register_node(id_to_tag, id_to_xz, bridge, point_a_id)
        node_b = register_node(id_to_tag, id_to_xz, bridge, point_b_id)

        elements.append(("truss", "cable", node_a, node_b, A, None, None))

# =============================================================================
# SOLVER
# =============================================================================
CABLE_MATERIAL_TAG = 200   # uniaxialMaterial tag namespace, separate from geomTransf tags

def run_solver(
    id_to_tag   : dict,
    id_to_xz    : dict,
    elements    : list,
    pin_tag     : int,
    roller_tag  : int,
    load_tags   : list[int],
    load_forces : list[float],
) -> bool:
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)
    ops.geomTransf('Linear', 1)

    # Register all nodes
    for point_id, tag in id_to_tag.items():
        x, z = id_to_xz[point_id]
        ops.node(tag, x, z)

    # Cable material — separate elastic modulus, referenced by truss elements
    ops.uniaxialMaterial('Elastic', CABLE_MATERIAL_TAG, IV.E_Tension)

    # Register all elements
    for eid, (kind, category, node_a, node_b, A, Iz, W) in enumerate(elements, start=1):
        if kind == "beam":
            ops.element('elasticBeamColumn', eid, node_a, node_b, A, IV.E_Steel, Iz, 1)
        elif kind == "truss":
            ops.element('Truss', eid, node_a, node_b, A, CABLE_MATERIAL_TAG)
        else:
            raise ValueError(f"Unknown element kind {kind!r} for eid {eid}")

    # Boundary conditions
    ops.fix(pin_tag,    1, 1, 0)
    ops.fix(roller_tag, 0, 1, 0)

    # Loads
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    for tag, force in zip(load_tags, load_forces):
        ops.load(tag, 0.0, force, 0.0)

    # Solve
    ops.system('BandGeneral')
    ops.numberer('RCM')
    ops.constraints('Plain')
    ops.integrator('LoadControl', 1.0)
    ops.algorithm('Linear')
    ops.analysis('Static')

    converged = (ops.analyze(1) == 0)
    return converged

# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================
def resolve_supports(bridge: BikeBridge, id_to_tag: dict, id_to_xz: dict) -> tuple[int, int]:
    pin_tag    = register_node(id_to_tag, id_to_xz, bridge, bridge.pin)
    roller_tag = register_node(id_to_tag, id_to_xz, bridge, bridge.roller)
    return pin_tag, roller_tag

# =============================================================================
# LOADS
# =============================================================================
def resolve_loads(bridge: BikeBridge, id_to_tag: dict, id_to_xz: dict) -> tuple[list[int], list[float]]:
    load_points = sorted(bridge.load_points, key=lambda point_id: bridge.points[point_id[0]][point_id[1]][0])
    n = len(load_points)

    x_vals = [float(bridge.points[point_id[0]][point_id[1]][0]) for point_id in load_points]

    deck_min, deck_max = IV.deck_range

    load_tags   = []
    load_forces = []

    for i in range(n):
        left_contrib  = (x_vals[i] - x_vals[i - 1]) / 2.0 if i > 0     else (x_vals[0] - deck_min) / 2.0
        right_contrib = (x_vals[i + 1] - x_vals[i]) / 2.0 if i < n - 1 else (deck_max - x_vals[-1]) / 2.0
        tributary_span = left_contrib + right_contrib

        Fz = IV.default_load * tributary_span * IV.tributary_width

        tag = register_node(id_to_tag, id_to_xz, bridge, load_points[i])
        load_tags.append(tag)
        load_forces.append(-Fz)

    return load_tags, load_forces

# =============================================================================
# OUTPUT UNIT CONVERSION
# =============================================================================
MM_TO_CM         = 0.1   # displacement: mm -> cm
STRESS_TO_KN_CM2 = 0.1   # stress: N/mm² -> kN/cm²

def convert_units(value: float | None, factor: float) -> float | None:
    return None if value is None else value * factor

# =============================================================================
# DISPLACEMENT RECOVERY
# =============================================================================
def max_displacement(id_to_tag: dict) -> float:
    peak = 0.0
    for tag in id_to_tag.values():
        ux, uz, _ = ops.nodeDisp(tag)
        magnitude = math.sqrt(ux**2 + uz**2)
        peak = max(peak, magnitude)
    return peak

# =============================================================================
# STRESS RECOVERY
# =============================================================================
def stress_scan(elements: list, category: str) -> tuple[float | None, float | None]:
    sig_max = None
    sig_min = None

    for eid, (kind, elem_category, node_a, node_b, A, Iz, W) in enumerate(elements, start=1):
        if elem_category != category:
            continue

        forces = ops.basicForce(eid)

        if kind == "beam":
            N, Mi, Mj = forces
            candidates = [
                N / A + Mi / W,
                N / A - Mi / W,
                N / A + Mj / W,
                N / A - Mj / W,
            ]
        elif kind == "truss":
            N, = forces
            candidates = [N / A]
        else:
            raise ValueError(f"Unknown element kind {kind!r} for eid {eid}")

        elem_max = max(candidates)
        elem_min = min(candidates)

        sig_max = elem_max if sig_max is None else max(sig_max, elem_max)
        sig_min = elem_min if sig_min is None else min(sig_min, elem_min)

    return sig_max, sig_min

# =============================================================================
# ORCHESTRATOR
# =============================================================================
def run_fea(bridge: BikeBridge) -> dict:
    elements  : list = []
    id_to_tag : dict = {}
    id_to_xz  : dict = {}

    build_frame_elements(bridge, id_to_tag, id_to_xz, elements)
    build_connector_elements(bridge, id_to_tag, id_to_xz, elements)
    build_cable_elements(bridge, id_to_tag, id_to_xz, elements)

    pin_tag, roller_tag       = resolve_supports(bridge, id_to_tag, id_to_xz)
    load_tags, load_forces    = resolve_loads(bridge, id_to_tag, id_to_xz)

    converged = run_solver(id_to_tag, id_to_xz, elements, pin_tag, roller_tag, load_tags, load_forces)

    if converged:
        frame_max,     frame_min     = stress_scan(elements, "frame")
        connector_max, connector_min = stress_scan(elements, "connector")
        cable_max,     cable_min     = stress_scan(elements, "cable")
        max_disp = max_displacement(id_to_tag)
    else:
        frame_max = frame_min = connector_max = connector_min = cable_max = cable_min = None
        max_disp = None

    return {
        "converged"        : converged,
        "max_displacement" : convert_units(max_disp, MM_TO_CM),
        "frame_stress"     : {
            "sig_max": convert_units(frame_max, STRESS_TO_KN_CM2),
            "sig_min": convert_units(frame_min, STRESS_TO_KN_CM2),
        },
        "connector_stress" : {
            "sig_max": convert_units(connector_max, STRESS_TO_KN_CM2),
            "sig_min": convert_units(connector_min, STRESS_TO_KN_CM2),
        },
        "cable_stress"      : {
            "sig_max": convert_units(cable_max, STRESS_TO_KN_CM2),
            "sig_min": convert_units(cable_min, STRESS_TO_KN_CM2),
        },
    }