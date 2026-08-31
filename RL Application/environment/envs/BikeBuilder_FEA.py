"""
FEA calculation for the terminal evaluation.
Built against BikeBridge — node identity is ID-based (frame_idx, local_idx),
not coordinate-based. No manual element splitting: BikeBridge.build_points
already spliced connector points into each tube's index sequence.

Beam formulation: toggled via IV.enable_timoshenko.
  True  -> ElasticTimoshenkoBeam (includes shear deformation, E+G+A+Iz+Avy)
  False -> elasticBeamColumn     (Euler-Bernoulli, no shear term, E+A+Iz)
Cables remain axial-only Truss elements regardless of this toggle.

Shear correction (Avy) is toggled via IV.enable_adaptive_shear.
  True  -> Cowper (1966) hollow-circle formula, using v = E/(2G) - 1 per material
  False -> flat IV.shear_correction_factor * A

Material inputs (E_frame, E_connection, E_cable, G_frames, G_connection)
are entered in kN/cm², matching Karamba3D's native convention, and
converted to N/mm² internally. Gravity/self-weight is tabled for now.
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
# UNIT CONVERSION CONSTANTS
# =============================================================================
KN_CM2_TO_N_MM2  = 10.0   # material inputs entered in kN/cm² -> N/mm² (used internally throughout this module)
MM_TO_CM         = 0.1    # output: displacement, mm -> cm
STRESS_TO_KN_CM2 = 0.1    # output: stress, N/mm² -> kN/cm²

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

def shear_area(A: float) -> float:
    # Effective shear area — flat approximation, IV.shear_correction_factor * A.
    return IV.shear_correction_factor * A

def poisson_from_E_G(E: float, G: float) -> float:
    # Derives Poisson's ratio from an isotropic material's E and G.
    # Unit-agnostic: E and G must share the same unit, but which one doesn't matter (ratio cancels).
    return E / (2.0 * G) - 1.0

def cowper_hollow_circle_shear_factor(outer_diameter: float, thickness: float, poisson_ratio: float) -> float:
    # Cowper (1966) shear correction factor for a hollow circular cross-section.
    # n = inner/outer radius ratio. Reduces to the standard solid-circle formula at n=0.
    inner_diameter = outer_diameter - 2.0 * thickness
    n  = inner_diameter / outer_diameter
    n2 = n * n
    numerator   = 6.0 * (1.0 + poisson_ratio) * (1.0 + n2)**2
    denominator = (7.0 + 6.0*poisson_ratio) * (1.0 + n2)**2 + (20.0 + 12.0*poisson_ratio) * n2
    return numerator / denominator

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
    poisson_frame = poisson_from_E_G(IV.E_frame, IV.G_frames) if IV.enable_adaptive_shear else None

    for property_name, tube_prefix in TUBE_PROPERTIES.items():
        per_frame_indices = getattr(bridge, property_name)

        for frame_idx, index_list in enumerate(per_frame_indices):
            stock_idx = bridge.stock_indices[frame_idx]
            row       = _crs_dataframe.iloc[stock_idx]

            outer_diameter = float(row[f"{tube_prefix}_OD"])
            thickness      = float(row[f"{tube_prefix}_T"])
            A, Iz = pipe_section(outer_diameter, thickness)
            W = Iz / (outer_diameter / 2.0)

            if IV.enable_adaptive_shear:
                shear_k = cowper_hollow_circle_shear_factor(outer_diameter, thickness, poisson_frame)
                Avy = shear_k * A
            else:
                Avy = shear_area(A)

            if IV.FEA_debug and tube_prefix == "SS":
                print(f"[SS] frame_idx={frame_idx:>2} stock_idx={stock_idx:>3} "
                      f"OD={outer_diameter:>8.3f} mm  t={thickness:>7.3f} mm  A={A:>10.3f} mm^2  "
                      f"Avy={Avy:>10.3f} mm^2 ({'cowper' if IV.enable_adaptive_shear else 'flat'})")

            for i in range(len(index_list) - 1):
                point_a_id = (frame_idx, index_list[i])
                point_b_id = (frame_idx, index_list[i + 1])

                node_a = register_node(id_to_tag, id_to_xz, bridge, point_a_id)
                node_b = register_node(id_to_tag, id_to_xz, bridge, point_b_id)

                elements.append(("beam", "frame", node_a, node_b, A, Iz, W, Avy))

# =============================================================================
# ELEMENT CONSTRUCTION — CONNECTORS
# =============================================================================
def build_connector_elements(bridge: BikeBridge, id_to_tag: dict, id_to_xz: dict, elements: list) -> None:
    A, Iz = pipe_section(IV.connector_OD, IV.connector_thickness)
    W = Iz / (IV.connector_OD / 2.0)

    if IV.enable_adaptive_shear:
        poisson_connection = poisson_from_E_G(IV.E_connection, IV.G_connection)
        shear_k = cowper_hollow_circle_shear_factor(IV.connector_OD, IV.connector_thickness, poisson_connection)
        Avy = shear_k * A
    else:
        Avy = shear_area(A)

    if IV.FEA_debug:
        print(f"[connector] OD={IV.connector_OD:.2f} t={IV.connector_thickness:.2f} "
              f"A={A:.3f} mm^2  Avy={Avy:.3f} mm^2 ({'cowper' if IV.enable_adaptive_shear else 'flat'})")

    for connection_set in bridge.connections:
        for entry in connection_set:
            for i in range(len(entry) - 1):
                point_a_id = entry[i]
                point_b_id = entry[i + 1]

                node_a = register_node(id_to_tag, id_to_xz, bridge, point_a_id)
                node_b = register_node(id_to_tag, id_to_xz, bridge, point_b_id)

                elements.append(("beam", "connector", node_a, node_b, A, Iz, W, Avy))

# =============================================================================
# ELEMENT CONSTRUCTION — CABLES
# =============================================================================
def build_cable_elements(bridge: BikeBridge, id_to_tag: dict, id_to_xz: dict, elements: list) -> None:
    A = solid_circle_area(IV.tension_OD)

    for point_a_id, point_b_id in bridge.tension_lines:
        node_a = register_node(id_to_tag, id_to_xz, bridge, point_a_id)
        node_b = register_node(id_to_tag, id_to_xz, bridge, point_b_id)

        # Avy is None: trusses are axial-only, shear deformation doesn't apply
        elements.append(("truss", "cable", node_a, node_b, A, None, None, None))

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
    ops.uniaxialMaterial('Elastic', CABLE_MATERIAL_TAG, IV.E_cable * KN_CM2_TO_N_MM2)

    # Per-category E and G, converted once, looked up during element registration
    E_by_category = {
        "frame"     : IV.E_frame      * KN_CM2_TO_N_MM2,
        "connector" : IV.E_connection * KN_CM2_TO_N_MM2,
    }
    G_by_category = {
        "frame"     : IV.G_frames     * KN_CM2_TO_N_MM2,
        "connector" : IV.G_connection * KN_CM2_TO_N_MM2,
    }

    # Register all elements
    for eid, (kind, category, node_a, node_b, A, Iz, W, Avy) in enumerate(elements, start=1):
        if kind == "beam":
            E = E_by_category[category]
            if IV.enable_timoshenko:
                G = G_by_category[category]
                ops.element('ElasticTimoshenkoBeam', eid, node_a, node_b, E, G, A, Iz, Avy, 1)
            else:
                ops.element('elasticBeamColumn', eid, node_a, node_b, A, E, Iz, 1)
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

    load_tags       = []
    load_forces     = []
    tributary_spans = []

    for i in range(n):
        left_contrib  = (x_vals[i] - x_vals[i - 1]) / 2.0 if i > 0     else (x_vals[0] - deck_min)
        right_contrib = (x_vals[i + 1] - x_vals[i]) / 2.0 if i < n - 1 else (deck_max - x_vals[-1])
        tributary_span = left_contrib + right_contrib
        tributary_spans.append(tributary_span)

        # NOTE: tributary_span is mm, tributary_width is m — deliberately NOT unit-matched.
        # The mm->m (÷1000) and kN->N (×1000) conversions cancel exactly, so this
        # multiplication correctly yields Newtons directly. Do not "fix" by adding
        # a conversion factor here.
        Fz = IV.default_load * tributary_span * IV.tributary_width

        tag = register_node(id_to_tag, id_to_xz, bridge, load_points[i])
        load_tags.append(tag)
        load_forces.append(-Fz)

    if IV.FEA_debug:
        print("\n--- resolve_loads ---")
        print(f"{'node':>6} {'x [mm]':>12} {'trib span [mm]':>16} {'Fz [N]':>14} {'Fz [kN]':>10}")
        for i in range(n):
            print(f"{load_tags[i]:>6} {x_vals[i]:>12.1f} {tributary_spans[i]:>16.1f} {load_forces[i]:>14.2f} {load_forces[i]/1000.0:>10.3f}")
        print(f"{'total':>6} {'':>12} {'':>16} {sum(load_forces):>14.2f} {sum(load_forces)/1000.0:>10.3f}\n")

    return load_tags, load_forces

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

    for eid, (kind, elem_category, node_a, node_b, A, Iz, W, Avy) in enumerate(elements, start=1):
        if elem_category != category:
            continue

        if kind == "beam":
            N1, V1, M1, N2, V2, M2 = ops.eleResponse(eid, 'localForce')
            N = -N1   # localForce's N1 reports tension as negative (opposite convention to basicForce/N2)
            if IV.enable_axial_only_stress:
                candidates = [N / A]
            else:
                candidates = [
                    N / A + M1 / W,
                    N / A - M1 / W,
                    N / A + M2 / W,
                    N / A - M2 / W,
                ]
        elif kind == "truss":
            N, = ops.basicForce(eid)
            candidates = [N / A]
        else:
            raise ValueError(f"Unknown element kind {kind!r} for eid {eid}")

        elem_max = max(candidates)
        elem_min = min(candidates)

        sig_max = elem_max if sig_max is None else max(sig_max, elem_max)
        sig_min = elem_min if sig_min is None else min(sig_min, elem_min)

    return sig_max, sig_min

# =============================================================================
# OUTPUT UNIT CONVERSION
# =============================================================================
def convert_units(value: float | None, factor: float) -> float | None:
    return None if value is None else value * factor

# =============================================================================
# REPORTING
# =============================================================================
def print_fea_result(result: dict) -> None:
    def fmt(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "—"

    print("\n=== FEA Result ===")
    print(f"{'Converged':<18}: {result['converged']}")
    print(f"{'Max displacement':<18}: {fmt(result['max_displacement'])} cm")
    print()
    print(f"{'Category':<12}{'sig_max [kN/cm2]':>20}{'sig_min [kN/cm2]':>20}")
    print("-" * 52)
    for label, key in (("Frame", "frame_stress"), ("Connector", "connector_stress"), ("Cable", "cable_stress")):
        sig_max = fmt(result[key]["sig_max"])
        sig_min = fmt(result[key]["sig_min"])
        print(f"{label:<12}{sig_max:>20}{sig_min:>20}")
    print("=" * 52 + "\n")

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

    if IV.FEA_debug:
        kind_counts     = {}
        category_counts = {}
        for kind, category, *_ in elements:
            kind_counts[kind]         = kind_counts.get(kind, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1

        print("\n--- model size ---")
        print(f"nodes   : {len(id_to_tag)}")
        print(f"elements: {len(elements)}  {dict(kind_counts)}  {dict(category_counts)}")
        print()


    pin_tag, roller_tag    = resolve_supports(bridge, id_to_tag, id_to_xz)
    load_tags, load_forces = resolve_loads(bridge, id_to_tag, id_to_xz)

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