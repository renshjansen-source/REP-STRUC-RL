# ASSISTING FUNCTIONS FOR THE MAIN_DEPLOYMENT FUNCTIONS

# Imports
from Grasshopper import DataTree
from Grasshopper.Kernel.Data import GH_Path
import Rhino.Geometry as rg

# Data Handlers | Renumber path structure
def renum_paths(input_tree):
    num_tree = DataTree[object]()

    for i in range(input_tree.BranchCount):
        branch   = input_tree.Branch(i)
        num_path = GH_Path(i)
        for item in branch:
            num_tree.Add(item, num_path)

    return num_tree

# Data Handlers | Geometrical copy functions
def point_copy(input_list):
    point_list = []
    for pt in input_list:
        new_point = rg.Point3d(pt.X, pt.Y, pt.Z)
        point_list.append(new_point)
    
    return point_list

# Start Planes 
def initial_targets(g_curve, first_frame, start_rotations):

    # Defining perpendicular placement plane
    start_tangent = g_curve.TangentAtStart
    start_normal  = rg.Vector3d(-start_tangent.Z, 0, start_tangent.X)
    start_point   = g_curve.PointAtStart
    start_plane   = rg.Plane(start_point, -start_normal, start_tangent)

    # Making additional rotations
    start_rotation_1 = rg.Transform.Rotation(start_rotations, start_plane.ZAxis, start_point)
    start_rotation_2 = rg.Transform.Rotation(-start_rotations, start_plane.ZAxis, start_point)
    start_plane_2    = rg.Plane(start_plane.Origin, start_plane.XAxis, start_plane.YAxis)
    start_plane_3    = rg.Plane(start_plane.Origin, start_plane.XAxis, start_plane.YAxis)
    start_plane_2.Transform(start_rotation_1)
    start_plane_3.Transform(start_rotation_2)

    # Pre-aligning first frame
    first_vector    = start_point - first_frame[4]
    first_ST_mid    = (first_frame[0] + first_frame[3]) / 2
    first_tangent   = first_ST_mid - first_frame[4]
    first_angle     = rg.Vector3d.VectorAngle(first_tangent, start_tangent, start_plane)

    first_translation = rg.Transform.Translation(first_vector)
    first_rotation    = rg.Transform.Rotation(first_angle, start_plane.ZAxis, start_point)
    first_compound    = first_translation * first_rotation

    
    for pt in first_frame:
        pt.Transform(first_translation)
        pt.Transform(first_rotation)
    
    # Retrieving orientation planes
    CS_tangent = first_frame[3] - first_frame[4]
    SS_tangent = first_frame[4] - first_frame[0]
    CS_normal  = rg.Vector3d(-CS_tangent.Z, 0, CS_tangent.X)
    SS_normal  = rg.Vector3d(-SS_tangent.Z, 0, SS_tangent.X)
    CS_mid     = (first_frame[3] + first_frame[4]) / 2
    SS_mid     = (first_frame[4] + first_frame[0]) / 2
    CS_plane   = rg.Plane(CS_mid, CS_tangent, CS_normal)
    SS_plane   = rg.Plane(SS_mid, SS_tangent, SS_normal)

    initial_planes = [start_plane, start_plane_2, start_plane_3, CS_plane, SS_plane]

    return initial_planes

def get_candidate_planes(stock_item, mirror):

    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    stock_planes = []

    for a, b in pairs:
        mid     = (stock_item[a] + stock_item[b]) / 2
        tangent = stock_item[a] - stock_item[b]
        normal  = rg.Vector3d(-tangent.Z, 0, tangent.X)

        if mirror == 0:
            plane = rg.Plane(mid, tangent, normal)
        else:
            plane = rg.Plane(mid, -tangent, normal)

        stock_planes.append(plane)

    # TT_mid     = (stock_item[0] + stock_item[1]) / 2
    # TT_tangent = stock_item[0] - stock_item[1]
    # TT_normal  = rg.Vector3d(-TT_tangent.Z, 0, TT_tangent.X)
    # TT_plane   = rg.Plane(TT_mid, TT_tangent, TT_normal)

    # HT_mid     = (stock_item[1] + stock_item[2]) / 2
    # HT_tangent = stock_item[1] - stock_item[2]
    # HT_normal  = rg.Vector3d(-HT_tangent.Z, 0, HT_tangent.X)
    # HT_plane   = rg.Plane(HT_mid, HT_tangent, HT_normal)

    # DT_mid     = (stock_item[2] + stock_item[3]) / 2
    # DT_tangent = stock_item[2] - stock_item[3]
    # DT_normal  = rg.Vector3d(-DT_tangent.Z, 0, DT_tangent.X)
    # DT_plane   = rg.Plane(DT_mid, DT_tangent, DT_normal)

    # CS_mid     = (stock_item[3] + stock_item[4]) / 2
    # CS_tangent = stock_item[3] - stock_item[4]
    # CS_normal  = rg.Vector3d(-CS_tangent.Z, 0, CS_tangent.X)
    # CS_plane   = rg.Plane(CS_mid, CS_tangent, CS_normal)

    # SS_mid     = (stock_item[4] + stock_item[0]) / 2
    # SS_tangent = stock_item[4] - stock_item[0]
    # SS_normal  = rg.Vector3d(-SS_tangent.Z, 0, SS_tangent.X)
    # SS_plane   = rg.Plane(SS_mid, SS_tangent, SS_normal)

    # stock_planes = [TT_plane, HT_plane, DT_plane, CS_plane, SS_plane]

    return stock_planes

def get_target_planes(stock_item, offset, mirror_flag):

    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    stock_planes = []

    for a, b in pairs:
        mid     = (stock_item[a] + stock_item[b]) / 2
        tangent = stock_item[a] - stock_item[b]
        normal  = rg.Vector3d(-tangent.Z, 0, tangent.X)

        if mirror_flag == False:
            plane   = rg.Plane(mid, -tangent, -normal)
            translation = rg.Transform.Translation(plane.YAxis * offset)
            plane.Transform(translation)
            stock_planes.append(plane)
        else:
            plane   = rg.Plane(mid, tangent, normal)
            translation = rg.Transform.Translation(plane.YAxis * offset)
            plane.Transform(translation)
            stock_planes.append(plane)



    return stock_planes