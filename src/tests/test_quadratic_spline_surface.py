"""
A lot of the ASOC code seems like deprecated things from the 6-split spline implementation of 
the quadratic surface.
"""


from src.core.common import *
from src.quadratic_spline_surface.quadratic_spline_surface import *
from src.utils.generate_shapes import *
from src.quadratic_spline_surface.optimize_spline_surface import *


def test_quadratic_reproduction(u_curvature: float = 0.0,
                                v_curvature: float = 0.0) -> bool:
    """
    Helper function that was commented out in the ASOC code.
    """
    resolution: int = 10
    param_grid: list[list[SpatialVector]]
    param_V: np.ndarray
    param_F: np.ndarray
    param_l: list[list[float]]
    control_point_grid: list[list[SpatialVector]]
    control_V: np.ndarray
    control_F: np.ndarray
    l: list[list[float]]
    face_to_patch_indices: list[list[int]]
    patch_to_face_indices: list[int]

    todo("I don't think this method is used that much anymore. As in, it's deprecated.")

    layout_point_grid: list[list[PlanarPoint]
                            ] = generate_global_layout_grid(resolution)
    control_point_grid = generate_quadratic_grid(layout_point_grid, u_curvature, v_curvature, 0)
    control_V, control_F, l = generate_mesh_from_grid(control_point_grid, True)
    param_grid = generate_plane_grid(resolution, 0.0, 0.0)
    param_V, param_F, param_l = generate_mesh_from_grid(param_grid, True)

    # quadratic_spline_params(spline_type=powell_sabin_six_split)
    optimization_params = OptimizationParameters()

    QuadraticSplineSurface()
