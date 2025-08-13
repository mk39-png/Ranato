"""
Methods to compute the contour curve network for a spline surface with view
frame.
"""

import time
import timeit
from dataclasses import dataclass
from enum import Enum
from timeit import Timer
from unittest.mock import patch

import igl
import numpy as np

from src.contour_network.compute_contours import \
    compute_spline_surface_contours_and_boundaries
from src.contour_network.compute_intersections import IntersectionParameters
from src.contour_network.intersection_data import IntersectionData
from src.contour_network.projected_curve_network import ProjectedCurveNetwork
from src.core.common import Matrix3x3f, PatchIndex, todo
from src.core.conic import Conic
from src.core.rational_function import RationalFunction
from src.quadratic_spline_surface.quadratic_spline_surface import \
    QuadraticSplineSurface


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

    def __init__(self,
                 spline_surface: QuadraticSplineSurface,
                 intersect_params: IntersectionParameters,
                 invisibility_params: InvisibilityParameters,
                 patch_boundary_edges: list[tuple[int, int]]) -> None:
        """
        Constructor that takes a spline surface and computes the full projected
        contour curve network with standard viewing frame along the z axis.
        :param spline_surface:       [in] quadratic spline surface to build contours for
        :param intersect_params:     [in] parameters for the intersection methods
        :param intersect_params:     [in] parameters for the invisibility methods
        :param patch_boundary_edges: [in] patch boundary edge indices (default none)
        """
        self.__init_contour_network(spline_surface,
                                    intersect_params,
                                    invisibility_params,
                                    patch_boundary_edges)

    def __init_contour_network(self,
                               spline_surface: QuadraticSplineSurface,
                               intersect_params: IntersectionParameters,
                               invisibility_params: InvisibilityParameters,
                               patch_boundary_edges: list[tuple[int, int]]):
        """
        Initialize the contour network
        """
        frame: Matrix3x3f = np.identity(3, dtype=np.float64)

        # Compute contours
        contour_domain_curve_segments: list[Conic]
        contour_segments: list[RationalFunction]  # <4, 3>
        contour_patch_indices: list[PatchIndex]
        contour_is_boundary: list[bool]
        contour_intersections: list[list[IntersectionData]]
        num_intersections: int

        time_start: float = timeit.default_timer()

        (contour_domain_curve_segments,
         contour_segments,
         contour_patch_indices,
         contour_is_boundary,
         contour_intersections,
         num_intersections) = compute_spline_surface_contours_and_boundaries(
            spline_surface,
            frame,
            patch_boundary_edges,
        )

        # Build contour labels for boundary
        time_end: float = timeit.default_timer()
