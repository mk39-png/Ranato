
import numpy as np

from src.contour_network.compute_intersections import (
    IntersectionParameters, compute_intersections,
    compute_planar_curve_intersections)
from src.contour_network.intersection_data import IntersectionData
from src.contour_network.intersection_heuristics import IntersectionStats
from src.core.common import (Matrix5x2f, Vector5f,
                             compare_list_list_varying_lengths,
                             compare_list_list_varying_lengths_float,
                             float_equal)
from src.core.rational_function import RationalFunction
from src.utils.compute_intersections_testing_utils import (
    compare_list_list_intersection_data,
    deserialize_list_list_intersection_data)
from src.utils.rational_function_testing_utils import \
    deserialize_rational_functions

# ********************
# Main Testing Methods
# ********************


def test_compute_intersections_spot_control() -> None:
    """
    Test compute_intersections() as it appears in init_contour_network() where 
    planar_contour_segments is another name for image_segments
    """
    filepath: str = "spot_control\\contour_network\\compute_intersections\\compute_intersections\\"
    planar_contour_segments: list[RationalFunction] = deserialize_rational_functions(
        filepath+"image_segments.json")
    intersect_params: IntersectionParameters = IntersectionParameters()
    contour_intersections: list[list[IntersectionData]] = deserialize_list_list_intersection_data(
        filepath+"contour_intersections_in.json")
    num_intersections_in: int = 0

    # Alias to match the method as it appears in init_contour_network
    intersection_knots: list[list[float]]
    intersection_indices: list[list[int]]
    intersection_call: int
    num_intersections_out: int

    (intersection_knots,
     intersection_indices,
     num_intersections_out,
     intersection_call) = compute_intersections(planar_contour_segments,
                                                intersect_params,
                                                contour_intersections,
                                                num_intersections_in)

    # Now initate comparisons
    contour_intersections_control: list[list[IntersectionData]] = (
        deserialize_list_list_intersection_data(filepath+"contour_intersections_out.json"))
    compare_list_list_varying_lengths_float(filepath+"intersections.csv", intersection_knots)
    compare_list_list_varying_lengths(filepath+"intersection_indices.csv", intersection_indices)
    # FXIME: contour_intersections are not as desired...
    compare_list_list_intersection_data(contour_intersections, contour_intersections_control)
    assert num_intersections_out == 183
    assert intersection_call == 378


def test_compute_spline_surface_boundary_intersections_spot_control() -> None:
    """

    """
    filepath: str = "spot_control\\contour_network\\compute_intersections\\compute_intersections\\"
    tester: list[list[IntersectionData]] = deserialize_list_list_intersection_data(
        filepath+"contour_intersections.json")


# *************************
# Original C++ test methods
# *************************

def test_compute_intersections_simple_linear_functions() -> None:
    """
    Simple Linear Functions.
    """
    first_P_coeffs: Matrix5x2f = np.array([[1, 0],
                                           [2, 1],
                                           [0, 0],
                                           [0, 0],
                                           [0, 0],])
    first_Q_coeffs: Vector5f = np.array([1, 0, 0, 0, 0])
    second_P_coeffs: Matrix5x2f = np.array([[4, 0],
                                            [-1, 1],
                                            [0, 0],
                                            [0, 0],
                                            [0, 0]])
    second_Q_coeffs: Vector5f = np.array([1, 0, 0, 0, 0])
    first_curve_intersections: list[float] = []
    second_curve_intersections: list[float] = []
    intersection_stats: IntersectionStats = IntersectionStats()
    intersection_params: IntersectionParameters = IntersectionParameters()

    first_image_segment = RationalFunction(4, 2, first_P_coeffs, first_Q_coeffs)
    second_image_segment = RationalFunction(4, 2, second_P_coeffs, second_Q_coeffs)
    compute_planar_curve_intersections(first_image_segment, second_image_segment,
                                       intersection_params,
                                       first_curve_intersections, second_curve_intersections,
                                       intersection_stats)
    assert len(first_curve_intersections) == 1
    assert len(second_curve_intersections) == 1
    float_equal(first_curve_intersections[0], 0.0)
    float_equal(second_curve_intersections[0], 0.0)
