"""
Class to build halfedge from VF

TODO The cleanest way to handle this is to fill boundaries with faces
and handle boundary cases with these to avoid invalid operations. The
interface should be chosen with care to balance elegance and versatility.

Mesh halfedge representation. Supports meshes with boundary and basic
topological information. Can be initialized from face topology information.
"""

from ..core.common import *


class HalfEdge:

    # ************
    # CONSTRUCTORS
    # ************
    def __init__(self):
        pass

    # ******
    # PUBLIC
    # ******
