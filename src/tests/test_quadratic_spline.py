from src.core.common import *
from src.core.apply_transformation import *
from src.core.generate_transformation import *
from src.core.compute_boundaries import *
from src.contour_network.contour_network import *
from src.quadratic_spline_surface.twelve_split_spline import *
from igl import readOBJ, writeOBJ

import logging
import sys
import argparse
import os


def test_main():
    # Build maps from strings to enums
    log_level_map: dict[str, int] = {
        "off": logging.NOTSET,
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARN,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }

    # Get command line arguments
    # parser = argparse.ArgumentParser(
    #     prog="ASOC",
    #     description="Generate smooth occluding contours for a mesh.",
    # )
    input_filename: str = "spot_control_mesh-cleaned_conf_simplified_with_uv.obj"
    output_dir: str = "./"
    log_level = logging.NOTSET
    color: Matrix3x1r = SKY_BLUE
    num_subdivisions: int = DISCRETIZATION_LEVEL
    optimization_params: OptimizationParameters = OptimizationParameters()
    weight: float = optimization_params.position_difference_factor

    # parser.add_argument('-i', '--input', required=True, help="Mesh filepath", type=str)
    # parser.add_argument('--log_level', help="Level of logging", action='store_const', type=int)
    # parser.add_argument("--num_subdivisions", help="Number of subdivisions", action='store_const', type=int)
    # parser.add_argument(
    #     '-w', '--weight', help="Fitting weight for the quadratic surface approximation", action='store_const', type=float)
    # args = parser.parse_args()

    # Set logger level
    logger.setLevel(log_level)

    # Set optimization parameters
    weight: float = optimization_params.position_difference_factor

    # Get input mesh
    V = np.ndarray(shape=(0, 0), dtype=np.float64)
    uv = np.ndarray(shape=(0, 0), dtype=np.float64)
    N = np.ndarray(shape=(0, 0), dtype=np.float64)
    F = np.ndarray(shape=(0, 0), dtype=np.int64)
    FT = np.ndarray(shape=(0, 0), dtype=np.int64)
    FN = np.ndarray(shape=(0, 0), dtype=np.int64)
    root_folder = os.getcwd()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    obj_dir = os.path.join(script_dir, input_filename)
    V, uv, N, F, FT, FN = igl.readOBJ(obj_dir)

    # Generate quadratic spline
    logger.info("Computing spline surface")
    face_to_patch_indices: list[list[int]] = []
    patch_to_face_indices: list[int] = []
    fit_matrix: csr_matrix
    energy_hessian: csr_matrix
    energy_hessian_inverse: CholeskySolverD

    # NOTE: must input a mesh that is already UV unwrapped....
    # TODO: but the specific UV unwrapping algorithm is special?
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)

    spline_surface: TwelveSplitSplineSurface = TwelveSplitSplineSurface(V, affine_manifold,  optimization_params)

    # View the mesh
    spline_surface.view(color, num_subdivisions)

    return

# if __name__ == "__main__":
#     main()
