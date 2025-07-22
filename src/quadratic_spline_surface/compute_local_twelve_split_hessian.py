"""
Used by optimize_spline_surface.py
"""

from src.core.common import Matrix2x2r
from src.quadratic_spline_surface.PS12_patch_coeffs import PS12_patch_coeffs

import numpy as np
import numpy.linalg as LA


def get_C_gl(uv: np.ndarray, corner_to_corner_uv_positions: list[Matrix2x2r], reverse_edge_orientations: list[bool]) -> np.ndarray:
    """
    Compute the matrix to convert from global degrees of freedom
    [f0, gu_0, gv_0, f1, gu_1, gv_1, f2, gu_2, gv_2, gm_12, gm_20, gm_01]
    to the local triangle degrees of freedom
    [f0, f1, f2, d01, d02, d10, d12, d20, d21, h01, h12, h20]

    Here, dij is the derivative of f in the direction of edge eij and
    hij is the derivative of f from the edge midpoint to the opposite vertex

    @param[in] uv: global uv vertex positions
    @param[in] corner_to_corner_uv_positions: per vertex matrices with the
    local vertex chart edge directions as rows
    @param[in] reverse_edge_orientations: per edge booleans indicating if the
    edge midpoint needs to be flipped from frame orientation consistency
    @return matrix mapping global to local degrees of freedom
    """

    assert uv.shape == (3, 2)
    assert len(corner_to_corner_uv_positions) == 3
    assert corner_to_corner_uv_positions[0].shape == (2, 2)
    assert len(reverse_edge_orientations) == 3

    # Get global uv vertex postions
    # TODO: apparently there's a difference betwen Vector and RowVector in Eigen which I'll need to translate...
    q0 = uv[[0], :]
    q1 = uv[[1], :]
    q2 = uv[[2], :]
    assert q0.shape == (1, 2)
    assert q1.shape == (1, 2)
    assert q2.shape == (1, 2)

    # Compute global edge directions and squared lengths
    e01_global = q1 - q0
    e12_global = q2 - q1
    e20_global = q0 - q2

    # TODO: is the below correct for dot product?
    l01sq: float = e01_global.dot(e01_global)
    l12sq: float = e12_global.dot(e12_global)
    l20sq: float = e20_global.dot(e20_global)
    assert isinstance(l01sq, float)

    # Compute global midpoint to corner directions
    e01_m = q2 - (q0 + q1) / 2
    e12_m = q0 - (q2 + q1) / 2
    e20_m = q1 - (q0 + q2) / 2

    # Compute global edge perpendicular directions
    # TODO: double check if perpendicular stuff is correct
    perpe01 = np.array([[-e01_global[0][1]], [e01_global[0][0]]])
    perpe12 = np.array([[-e12_global[0][1]], [e12_global[0][0]]])
    perpe20 = np.array([[-e20_global[0][1]], [e20_global[0][0]]])
    assert perpe01.shape == (1, 2)

    # Compute local edge midpoint directions in frames defined by eij, perpeij
    e01_m_loc = np.array([[e01_m.dot(e01_global) / l01sq],
                          [e01_m.dot(perpe01) / l01sq]])
    e12_m_loc = np.array([[e12_m.dot(e12_global) / l12sq],
                          [e12_m.dot(perpe12) / l12sq]])
    e20_m_loc = np.array([[e20_m.dot(e20_global) / l20sq],
                          [e20_m.dot(perpe20) / l20sq]])
    assert e01_m_loc.shape == (1, 2)

    # Extract vertex chart rotated corner to corner uv directions
    # Note that eij = R(qj - qi), where R is some rigid transformation mapping
    # the global uv triangle to some local layout containing the given corner
    e01: np.ndarray = corner_to_corner_uv_positions[0][[0], :]  # q1 - q0
    e02: np.ndarray = corner_to_corner_uv_positions[0][[1], :]  # q2 - q0
    e12: np.ndarray = corner_to_corner_uv_positions[1][[0], :]  # q2 - q1
    e10: np.ndarray = corner_to_corner_uv_positions[1][[1], :]  # q0 - q1
    e20: np.ndarray = corner_to_corner_uv_positions[2][[0], :]  # q0 - q2
    e21: np.ndarray = corner_to_corner_uv_positions[2][[1], :]  # q1 - q2
    assert e01.shape == (1, 2)

    # Assign labels to the elementary row basis vectors in R^12 corresponding
    # to global degree of freedom
    f0: np.ndarray = np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
    gu_0: np.ndarray = np.array([[0], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
    gv_0: np.ndarray = np.array([[0], [0], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
    f1: np.ndarray = np.array([[0], [0], [0], [1], [0], [0], [0], [0], [0], [0], [0], [0]])
    gu_1: np.ndarray = np.array([[0], [0], [0], [0], [1], [0], [0], [0], [0], [0], [0], [0]])
    gv_1: np.ndarray = np.array([[0], [0], [0], [0], [0], [1], [0], [0], [0], [0], [0], [0]])
    f2: np.ndarray = np.array([[0], [0], [0], [0], [0], [0], [1], [0], [0], [0], [0], [0]])
    gu_2: np.ndarray = np.array([[0], [0], [0], [0], [0], [0], [0], [1], [0], [0], [0], [0]])
    gv_2: np.ndarray = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [1], [0], [0], [0]])
    gm_12: np.ndarray = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [0], [0]])
    gm_20: np.ndarray = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [0]])
    gm_01: np.ndarray = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1]])
    assert f0.shape == (12, 1)

    # Compute the derivative of f in the directions of the parametric edges eij
    # Note that these (and subsequent quantities) are actually row vectors that extract
    # the desired values from the column vector of global degrees of freedom by multiplication
    d01: np.ndarray = e01[0][0] * gu_0 + e01[0][1] * gv_0
    d02: np.ndarray = e02[0][0] * gu_0 + e02[0][1] * gv_0
    d10: np.ndarray = e10[0][0] * gu_1 + e10[0][1] * gv_1
    d12: np.ndarray = e12[0][0] * gu_1 + e12[0][1] * gv_1
    d20: np.ndarray = e20[0][0] * gu_2 + e20[0][1] * gv_2
    d21: np.ndarray = e21[0][0] * gu_2 + e21[0][1] * gv_2
    assert d01.shape == (12, 1)

    # Cmpute the derivative along eij at the edge midpoint
    ge_01: np.ndarray = -2 * f0 - 0.5 * d01 + 2 * f1 + 0.5 * d10
    ge_12: np.ndarray = -2 * f1 - 0.5 * d12 + 2 * f2 + 0.5 * d21
    ge_20: np.ndarray = -2 * f2 - 0.5 * d20 + 2 * f0 + 0.5 * d02
    assert ge_01.shape == (12, 1)

    # Reverse (global) edge orientations as needed for consistency of the
    # midpoint reference frames
    if (reverse_edge_orientations[0]):
        gm_12 = -gm_12
    if (reverse_edge_orientations[1]):
        gm_20 = -gm_20
    if (reverse_edge_orientations[2]):
        gm_01 = -gm_01

    # Compute the derivative of f in the direction of the edge midpoint to opposite
    # vertex
    h01: np.ndarray = e01_m_loc[0] * ge_01 + e01_m_loc[1] * gm_01
    h12: np.ndarray = e12_m_loc[0] * ge_12 + e12_m_loc[1] * gm_12
    h20: np.ndarray = e20_m_loc[0] * ge_20 + e20_m_loc[1] * gm_20
    assert h01.shape == (12, 1)

    # Assemble matrix
    # TODO: check if slicing is correct
    C_gl: np.ndarray = np.array([
        f0[:, 0],
        f1[:, 0],
        f2[:, 0],
        d01[:, 0],
        d02[:, 0],
        d10[:, 0],
        d12[:, 0],
        d20[:, 0],
        d21[:, 0],
        h01[:, 0],
        h12[:, 0],
        h20[:, 0]
    ])
    assert C_gl.shape == (12, 12)

    return C_gl


def get_Tquad_Cder_Csub() -> np.ndarray:
    """
    Compute the matrix to go from local triangle degrees of freedom
    [f0, f1, f2, d01, d02, d10, d12, d20, d21, h01, h12, h20]
    to the second derivatives
    [paa, pbb, pab]_{1,...,12}
    of the twelve subtriangle quadratic functions defined by the Powell-
    Sabin interpolant with respect to the barycentric coordinates of
    the domain triangle

    The full chain of theoretical operations is:
        Csub: local triangle dof -> Bezier points
        Cder: Bezier points -> second derivatives in subtriangle coords
        Tquad: second derivatives in subtriangle coords
            -> second derivatives in triangle coords
    However, this matrix is constant and is thus hard coded

    @return matrix mapping local dof to second derivatives 
    """
    # Autogenerate matrix entries as 12 subtriangle matrices
    # Note that unused values (first derivatives and constants) are also generated
    # but are unused
    patch_coeffs: np.ndarray = PS12_patch_coeffs()

    # Combine 12 subtriangle matrices, flattening the second derivatives per subtriangle
    # TODO: could use numpy indexing somehow or some sort of flattening feature for optimization
    Tquad_Cder_Csub: np.ndarray = np.ndarray(shape=(36, 12), dtype=np.float64)
    for i in range(12):
        for j in range(3):
            for k in range(12):
                Tquad_Cder_Csub[i * 3 + j, k] = patch_coeffs[i][j + 3][k]

    return Tquad_Cder_Csub


def get_R_quad(uv: np.ndarray) -> np.ndarray:
    """
    Compute the matrix to go from second derivatives 
    [paa, pbb, pab]_{1,...,12}
    of the twelve subtriangle quadratic functions defined by the Powell-
    Sabin interpolant with respect to the barycentric coordinates of
    the domain triangle to the second derivatives with respect to a 
    parametric domain triangle defined by uv

    @param[in] uv: uv coordinates of the domain triangle
    @return matrix mapping second derivatives wrt barycentric coordinates
        to second derivatives wrt uv coordinates
    """
    assert uv.shape == (3, 2)
    # Compute 2x2 matrix mapping parametric to barycentric coordinates
    # TODO: ensure that this is the same as Eigen matrix construction
    R_inv: np.ndarray = np.array(
        [[uv[0, 0] - uv[2, 0], uv[1, 0] - uv[2, 0]],
         [uv[0, 1] - uv[2, 1], uv[1, 1] - uv[2, 1]]])
    assert R_inv.shape == (2, 2)

    # Compute inverse 2x2 matrix mapping barycentric to parametric coordinates
    R: np.ndarray = np.array(
        [[R_inv[1, 1] - R_inv[0, 1]],
         [-R_inv[1, 0], R_inv[0, 0]]])
    R = R / (R_inv[0, 0] * R_inv[1, 1] - R_inv[0, 1] * R_inv[1, 0])
    assert R.shape == (2, 2)

    # Compute 3x3 matrix q_l mapping second derivatives wrt barycentric coordinates
    # to second derivatives wrt parametric coordinates
    q_l: np.ndarray = np.array([
        [R[0, 0] * R[1, 1] + R[0, 1] * R[1, 0], 2 * R[0, 0] * R[0, 1], 2 * R[1, 0] * R[1, 1]],
        [2 * R[0, 0] * R[1, 0], 2 * R[0, 0] * R[0, 0], 2 * R[1, 0] * R[1, 0]],
        [2 * R[0, 1] * R[1, 1], 2 * R[0, 1] * R[0, 1], 2 * R[1, 1] * R[1, 1]]
    ])
    assert q_l.shape == (3, 3)

    # Build R_quad as 12 block copies of q_l
    # TODO: could use NumPy indexing somehow
    R_quad: np.ndarray = np.zeros(shape=(36, 36))
    for i in range(12):
        for j in range(3):
            for k in range(3):
                R_quad[i * 3 + j, i * 3 + k] = q_l[j, k]

    return R_quad


def get_S_weighted(A: float) -> np.ndarray:
    """
    Get barycentric patch triangle area matrix weighted by some arbitrary area A

    Return np.ndarray of shape (36, 36)
    """
    S: np.ndarray = np.zeros(shape=(36, 36), dtype=float)

    # TODO: optimize with numpy indexing
    for i in range(36):
        if (i % 3) == 0:
            S[i, i] = 2 * A
        else:
            S[i, i] = A

    for i in range(18):
        S[i, i] = S[i, i] / 24

    for i in range(18, 36):
        S[i, i] = S[i, i] / 8

    return S


def get_S(uv: np.ndarray) -> np.ndarray:
    """
    Compute the diagonal uv triangle area weighting matrix for the Hessian

    @param[in] uv: uv coordinates of the domain triangle
    @return diagonal Hessian weight matrix
    """
    # Compute the area of the uv triangle
    tri: np.ndarray = np.array(
        [[uv[0, 0], uv[0, 1], 1],
         [uv[1, 0], uv[1, 1], 1],
         [uv[2, 0], uv[2, 1], 1]])
    A: float = LA.det(tri) / 2

    # Compute the corresponding Hessian weighting matrix
    return get_S_weighted(A * A)


def build_local_smoothness_hessian(uv: np.ndarray, corner_to_corner_uv_positions: list[Matrix2x2r], reverse_edge_orientations: list[bool]) -> np.ndarray:
    """
    Compute the thin plate energy Hessian matrix for a single 12-split 
    Powell-Sabin element

    @param[in] uv: uv coordinates of the domain triangle
    @param[in] corner_to_corner_uv_positions: per vertex matrices with the
    local vertex chart edge directions as rows
    @param[in] reverse_edge_orientations: per edge booleans indicating if the
    edge midpoint needs to be flipped from frame orientation consistency
    @return Hessian matrix in terms of Powell-Sabin degrees of freedom
    """
    #  Build all elementary matrices composing the Hessian
    C_gl: np.ndarray = get_C_gl(uv, corner_to_corner_uv_positions, reverse_edge_orientations)
    assert C_gl.shape == (12, 12)
    Tquad_Cder_Csub: np.ndarray = get_Tquad_Cder_Csub()
    assert Tquad_Cder_Csub.shape == (36, 12)
    R_quad: np.ndarray = get_R_quad(uv)
    assert R_quad.shape == (36, 36)
    S: np.ndarray = get_S(uv)
    assert S.shape == (36, 36)

    # Assemble matrices into the Hessian
    # G shape (36, 12) = (36, 36) @ (36, 12) @ (12, 12)
    G: np.ndarray = R_quad @ Tquad_Cder_Csub @ C_gl
    assert G.shape == (36, 12)

    # hessian shape (12, 12) = (12, 36) @ (36, 36) @ (36, 12)
    hessian: np.ndarray = G.T @ S @ G
    assert hessian.shape == (12, 12)

    return hessian
