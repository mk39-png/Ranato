"""
Methods to compute the contour curve network for a spline surface with view
frame.
"""

from dataclasses import dataclass
from enum import Enum

from src.contour_network.projected_curve_network import ProjectedCurveNetwork
from src.core.common import todo


class InvisibilityMethod(Enum):
    NONE = 0  # Set all QI to 0
    DIRECT = 1    # Ray test per segment
    CHAINING = 2    # Ray test per chain of segments between features
    PROPAGATION = 3    # Ray test for connected components with local propagation


@dataclass
class InvisibilityParameters():
    """
    Parameters for the invisibility computation
    """
    pad_amount: float = 1e-9  # Padding for contour domains
    write_contour_soup = False  # Option to write contours before graph construction for diagnostics

    # Method for computing quantitative visibility
    invisibility_method: InvisibilityMethod = InvisibilityMethod.CHAINING

    # Options to view each local propagation step during computation for debugging
    view_intersections = False
    view_cusps = False

    # Options for redundancy checks
    poll_chain_segments = True  # Sample and poll 3 segments for majority per chain QI
    poll_segment_points = False  # Sample and poll 3 points for majority per segment QI

    # Consistency checks
    check_chaining = False
    check_propagation = False


class ContourNetwork(ProjectedCurveNetwork):
    """
    @brief Class to compute the projected contours of a quadratic spline surface
    and represent them as a curve network. Also computes the quantitative
    invisibility for the contours.
    """
    def __init__() -> None:
        """
        Constructor that takes a spline surface and computes the full projected
        contour curve network with standard viewing frame along the z axis.
        :param spline_surface:       [in] quadratic spline surface to build contours for
        :param intersect_params:     [in] parameters for the intersection methods
        :param intersect_params:     [in] parameters for the invisibility methods
        :param patch_boundary_edges: [in] patch boundary edge indices (default none)
        """
        todo()
