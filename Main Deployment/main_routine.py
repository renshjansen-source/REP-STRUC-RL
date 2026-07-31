# IMPORTS | LINKING
import sys
sys.path.append(r"C:\Users\rensh\Documents\Master Thesis\REP_STRUC RL\Main Deployment")

import Rhino.Geometry as rg
import importlib
from math import pi
import gh_assist

importlib.reload(gh_assist)

from gh_assist import point_copy, initial_targets, get_candidate_planes, get_target_planes

# INTERNAL VARIABLES
start_rotations = 0.25 * pi # Allows for additional rotating perpendicular to the start of the curve
con_offset      = 50.0      # EXTERNALIZE

# MAIN FUNCTIONS

def MAIN(g_curve, stock_selections, shape_grammars):

    moved_stock     = []
    check_targets   = []

    # Retrieving the first frames
    first_frame     = point_copy(stock_selections[0])
    target_planes   = initial_targets(g_curve, first_frame, start_rotations)

    # Main Loop
    for i, grammar in enumerate(shape_grammars):
        check_targets.append(target_planes)

        # Seperating the grammar
        target_idx    = grammar[1]
        candidate_idx = grammar[2]
        mirror_idx    = grammar[3]

        # Retrieving candidate plane
        candidate_planes = get_candidate_planes(stock_selections[i], mirror_idx)

        # Case 1: Given state is user defined
        picked_target    = target_planes[target_idx]
        picked_candidate = candidate_planes[candidate_idx]

        frame  = point_copy(stock_selections[i])
        orient = rg.Transform.PlaneToPlane(picked_candidate, picked_target)
        for pt in frame:
            pt.Transform(orient)
        
        moved_stock.append(frame)
        
        # Setting variables for next iteration
        if mirror_idx == 0:
            mirror_flag = False
        else:
            mirror_flag = True
        
        target_planes = get_target_planes(frame, con_offset, mirror_flag)
        

    return moved_stock, check_targets


# MAIN FUNCTIONS | PLACEMENT ROUTINE