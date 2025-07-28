"""
Methods to evaluate normals to a quadratic surface with Zwart-Powell basis
coefficients.
"""
from src.core.bivariate_quadratic_function import u_derivative_matrix, v_derivative_matrix, compute_quadratic_cross_product
from src.core.common import Matrix6x3r
import numpy as np

# TODO: add typing here for shape and whatnot surface


def generate_quadratic_surface_normal_coeffs(surface_mapping_coeffs) -> Matrix6x3r:
    """
    Compute the quadratic coefficients of the normal vector to a quadratic
    surface.

    @param[in] surface_mapping_coeffs: coefficients for the quadratic surface

    @return Coefficients for the quadratic polynomial defining the normal vector
    on the surface
    """
    # Get directional derivatives
    D_u: np.ndarray = u_derivative_matrix()
    D_v: np.ndarray = v_derivative_matrix()
    # TODO: double check matmul
    u_derivative_coeffs = D_u @ surface_mapping_coeffs
    v_derivative_coeffs = D_v @ surface_mapping_coeffs
    assert D_u.shape == (3, 6)
    assert D_v.shape == (3, 6)
    assert u_derivative_coeffs.shape == (3, 3)
    assert v_derivative_coeffs.shape == (3, 3)

    # Compute normal from the cross product
    normal_mapping_coeffs = compute_quadratic_cross_product(
        u_derivative_coeffs, v_derivative_coeffs)

    return normal_mapping_coeffs
