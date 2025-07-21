"""
powell_sabin_local_to_global_indexing.py
Methods to generate local to global indexing maps for six and twelve split
Powell-Sabin spline surfaces.

Used by optimize_spline_surface.py
"""

from typing import Literal
from src.core.common import VectorX
from src.core.affine_manifold import *
from src.core.differentiable_variable import *
from src.core.halfedge import *
from src.quadratic_spline_surface.position_data import *

import numpy as np

# TODO: determine what the translation of autodiff.h Gradient and Hessian would be in NumPy.
Gradient = np.ndarray
Hessian = np.ndarray

# The variables we are optimizing need to be linearized and initialized with
# some values. Similarly, the final optimized results need to be extracted.
# Furthermore, for the optimization it is useful to combine the variable values
# into a single vector so that, e.g., gradient descent or Newton's method can
# be applied.

# *******************
# Local block indices
# *******************


def generate_local_vertex_position_variables_start_index(vertex_index: int, dimension: int = 3) -> int:
    """
    Get the local start index of the block of position variable indices for a
    given vertex
    """
    relative_vertex_index = 3 * dimension * vertex_index
    return relative_vertex_index


def generate_local_vertex_gradient_variables_start_index(vertex_index: int, dimension: int = 3) -> int:
    """
    Get the local start index of the block of gradient variable indices for a
    given vertex
    """
    relative_vertex_index: int = 3 * dimension * vertex_index
    position_block_size: int = dimension
    return relative_vertex_index + position_block_size


def generate_local_edge_gradient_variables_start_index(edge_index: int, dimension: int = 3) -> int:
    """ 
    Get the local start index of the block of gradient variable indices for a
    given edge
    """
    vertex_block_size: int = 9 * dimension
    relative_edge_index = dimension * edge_index

    return vertex_block_size + relative_edge_index

# **********************
# Local variable indices
# **********************


def generate_local_vertex_position_variable_index(face_vertex_index: int, coord: int, dimension: int = 3) -> int:
    """
    Used in optimize_spline_surface.py

    Compute the index of a vertex position variable in a local DOF vector. 
    i.e. Get the local index of the position variable indices for a given coordinate
    and vertex index

    @param[in] face_vertex_index: index of the vertex in the face
    @param[in] coord: coordinate of the variable
    @param[in] dimension: number of coordinate dimensions
    @return index of the variable in the local DOF vector
    """
    start_index = generate_local_vertex_position_variables_start_index(face_vertex_index, dimension)

    return start_index + coord


def generate_local_vertex_gradient_variable_index(face_vertex_index: int,
                                                  row: int, col: int, dimension: int = 3) -> int:
    """
    Used in optimize_spline_surface.py

    Compute the index of a vertex gradient variable in a local DOF vector.
    i.e. Get the local index of the gradient variable indices for a given matrix index
    pair and vertex index

    @param[in] face_vertex_index: index of the vertex in the face
    @param[in] row: row of the gradient matrix variable
    @param[in] col: column of the gradient matrix variable
    @param[in] dimension: number of coordinate dimensions
    @return index of the variable in the local DOF vector
    """
    start_index: int = generate_local_vertex_gradient_variables_start_index(face_vertex_index, dimension)
    matrix_index: int = generate_local_variable_matrix_index(row, col, dimension)

    return start_index + matrix_index


def generate_local_edge_gradient_variable_index(face_edge_index: int, coord: int, dimension: int = 3) -> int:
    """
    Used in optimize_spline_surface.py

    Compute the index of a edge gradient variable in a local DOF vector.
    i.e. Get the local index of the gradient variable indices for a given coordinate
    and edge index pair

    @param[in] face_vertex_index: index of the edge in the face
    @param[in] coord: coordinate of the variable
    @param[in] dimension: number of coordinate dimensions
    @return index of the variable in the local DOF vector
    """

    start_index: int = generate_local_edge_gradient_variables_start_index(face_edge_index, dimension)

    return start_index + coord


# ********************
# Global block indices
# ********************

def generate_global_vertex_position_variables_block_start_index() -> Literal[0]:
    """Get the start index of the block of vertex position variable indices"""
    return 0


def generate_global_vertex_gradient_variables_block_start_index(num_variable_vertices: int, dimension: int) -> int:
    """Get the start index of the block of vertex gradient variable indices"""

    # There are dimension many position variables per variable vertex
    return dimension * num_variable_vertices


def generate_global_edge_gradient_variables_block_start_index(num_variable_vertices: int, dimension: int) -> int:
    """ Get the start index of the block of edge gradient variable indices"""
    # There are dimension many position variables and 2 * dimension many vector
    # gradient variables per variable vertex
    return 3 * dimension * num_variable_vertices


def generate_global_vertex_position_variables_start_index(vertex_index: int, dimension: int) -> int:
    """
    Get the start index of the block of position variable indices for a given
    vertex
    """
    start_index: int = generate_global_vertex_position_variables_block_start_index()
    relative_vertex_index: int = dimension * vertex_index

    return start_index + relative_vertex_index


def generate_global_vertex_gradient_variables_start_index(num_variable_vertices: int, vertex_index: int, dimension: int) -> int:
    """
    Get the start index of the block of gradient variable indices for a given
    vertex
    """
    start_index: int = generate_global_vertex_gradient_variables_block_start_index(num_variable_vertices, dimension)
    relative_vertex_index: int = 2 * dimension * vertex_index

    return start_index + relative_vertex_index


def generate_global_edge_gradient_variables_start_index(num_variable_vertices: int,
                                                        edge_index: int,
                                                        dimension: int) -> int:
    """
    Get the start index of the block of gradient variable indices for a given
    edge
    """
    start_index: int = generate_global_edge_gradient_variables_block_start_index(num_variable_vertices, dimension)
    relative_edge_index: int = dimension * edge_index

    return start_index + relative_edge_index


# ***********************
# Global variable indices
# ***********************

def generate_global_vertex_position_variable_index(vertex_index: int, coord: int, dimension: int = 3) -> int:
    """
    Used locally.

    Compute the index of a vertex position variable in a global DOF vector.
    i.e. Get the global index of the position variable indices for a given coordinate
    and vertex index

    @param[in] vertex_index: index of the vertex in the mesh
    @param[in] coord: coordinate of the variable
    @param[in] dimension: number of coordinate dimensions
    @return index of the variable in the global DOF vector
    """
    start_index: int = generate_global_vertex_position_variables_start_index(vertex_index, dimension)

    return start_index + coord


def generate_global_vertex_gradient_variable_index(num_variable_vertices: int,
                                                   vertex_index: int,
                                                   row: int,
                                                   col: int,
                                                   dimension: int = 3) -> int:
    """
    Used locally.

    Compute the index of a vertex gradient variable in a global DOF vector.
    i.e. Get the index of the gradient variable indices for a given matrix index pair
    and vertex index

    @param[in] num_variable_vertices: number of variable vertices for the optimization
    @param[in] vertex_index: index of the vertex in the mesh
    @param[in] row: row of the gradient matrix variable
    @param[in] col: column of the gradient matrix variable
    @param[in] dimension: number of coordinate dimensions
    @return index of the variable in the global DOF vector
    """
    start_index: int = generate_global_vertex_gradient_variables_start_index(
        num_variable_vertices, vertex_index, dimension)
    matrix_index: int = generate_local_variable_matrix_index(row, col, dimension)

    return start_index + matrix_index


def generate_global_edge_gradient_variable_index(num_variable_vertices: int,
                                                 edge_index: int,
                                                 coord: int,
                                                 dimension: int = 3) -> int:
    """
    Used locally.

    Compute the index of an edge gradient variable in a global DOF vector.
    i.e. Get the index of the gradient variable indices for a given coordinate and
    edge index pair

    @param[in] num_variable_vertices: number of variable vertices for the optimization
    @param[in] edge_index: index of the edge in the mesh
    @param[in] coord: coordinate of the variable
    @param[in] dimension: number of coordinate dimensions
    @return index of the variable in the global DOF vector
    """
    start_index: int = generate_global_edge_gradient_variables_start_index(num_variable_vertices, edge_index, dimension)

    return start_index + coord


# *******************
# Variable flattening
# *******************

def generate_six_split_variable_value_vector(vertex_positions: list[SpatialVector], vertex_gradients: list[Matrix2x3r], variable_vertices: list[int]) -> VectorX:
    """"
    Given vertex positions and gradients and a list of variable vertices, assemble
    the vector of global vertex degrees of freedom

    This is the complete list of degrees of freedom for the six split, and it is
    a subset of the degrees of freedom for the twelve split.

    i.e. Get flat vector of all current variable values for the six-split
    NOTE: Also used as a subroutine to generate the twelve split maps

    @param[in] vertex_positions: list of vertex position values
    @param[in] vertex_gradients: list of vertex gradient matrices 
    @param[in] variable_vertices: list of variable vertex indices
    @param[out] variable_values: vertex DOF vector
    """

    num_variable_vertices = len(variable_vertices)
    if num_variable_vertices == 0:
        logger.warning("Building value vector for zero variable vertices")
        raise Exception("Does this ever happen?")
        return

    dimension: int = 3

    # FIXME: right now, variable_values is shape (n, )... may want to change that so it's at least (n, 1) or (1, n) depending on its use case
    variable_values: VectorX = np.ndarray(shape=(3 * dimension * num_variable_vertices, ))

    # Get postion values
    for vertex_index in range(num_variable_vertices):
        start_index: int = generate_global_vertex_position_variables_start_index(
            vertex_index, dimension)

        for i in range(dimension):
            # NOTE: SpatialVector shape is (1, 3)
            # FIXME: something might go wrong with vertex_positions accessing because of SpatialVector shape
            variable_values[start_index + i] = vertex_positions[variable_vertices[vertex_index]][0, i]

    # Get gradient values
    for vertex_index in range(num_variable_vertices):
        start_index: int = generate_global_vertex_gradient_variables_start_index(
            num_variable_vertices, vertex_index, dimension)
        variable_matrix: Matrix2x3r = vertex_gradients[variable_vertices[vertex_index]]
        # TODO: change below to use numpy indexing?
        for i in range(variable_matrix.shape[0]):  # rows
            for j in range(variable_matrix.shape[1]):  # columns
                local_index: int = generate_local_variable_matrix_index(i, j, dimension)
                variable_values[start_index + local_index] = variable_matrix[i, j]

    return variable_values


def generate_twelve_split_variable_value_vector(vertex_positions: list[SpatialVector], vertex_gradients: list[Matrix2x3r], edge_gradients: list[list[SpatialVector]], variable_vertices: list[int], variable_edges: list[int], halfedge: Halfedge, he_to_corner: list[tuple[Index, Index]]) -> VectorX:
    """
    Used in optimize_spline_surface.py

    Given vertex positions and gradients, edge gradients, and lists
    of variable vertices and edges assemble the vector of global
    degrees of freedom for the twelve split
    i.e. Get flat vector of all current variable values for the twelve-split

    @param[in] vertex_positions: list of vertex position values
    @param[in] vertex_gradients: list of vertex gradient matrices 
    @param[in] edge_gradients: list of edge gradient normal vectors 
    @param[in] variable_vertices: list of variable vertex indices
    @param[in] variable_edges: list of variable edge indices
    @param[in] halfedge: halfedge data structure
    @param[in] he_to_corner: map from halfedges to opposite triangle corners
    @param[out] variable_values: twelve-split DOF vector
    """
    #  Get the variable values shared with the six-split
    # TODO: wait, can the below be None or no?
    six_split_variable_values: VectorX = generate_six_split_variable_value_vector(
        vertex_positions, vertex_gradients, variable_vertices)

    #  Add six split to the variable value vector
    #  Build a halfedge representation to get unique edge values
    num_variable_vertices: int = len(variable_vertices)
    num_variable_edges: int = len(variable_edges)
    dimension: int = 3

    # TODO: address the face that VectorX here is shape (n,) rather than (n,1) like in the other files....
    variable_values: VectorX = np.ndarray(shape=(3 * dimension * num_variable_vertices +
                                                 dimension * num_variable_edges,))
    variable_values[:six_split_variable_values.size] = six_split_variable_values

    # Get flat values for edge gradients
    for variable_edge_index in range(num_variable_edges):
        # Get one corner for the given edge
        edge_index: Index = variable_edges[variable_edge_index]
        halfedge_index: Index = halfedge.edge_to_first_halfedge(edge_index)
        face_index: Index = he_to_corner[halfedge_index][0]
        face_vertex_index: Index = he_to_corner[halfedge_index][1]

        # Extract each coordinate for the corner
        num_coordinates: int = 3
        for coord in range(num_coordinates):
            # Get edge variable index
            variable_index: int = generate_global_edge_gradient_variable_index(
                num_variable_vertices, variable_edge_index, coord)

            # Extract variable value of the edge to the corner
            variable_values[variable_index] = edge_gradients[face_index][face_vertex_index][coord]

    # TODO: address the shaping of variable values, which is (n,) right now
    # i.e. it might need to be (1, n) or (n, 1)
    return variable_values


def generate_six_split_local_to_global_map(global_vertex_indices: list[int], num_variable_vertices: int) -> list[int]:
    """
    Given the global vertex indices of a triangle, compute the map from the
    local DOF vector indices for this triangle to their indices in the global
    DOF vector for the six-split

    This is used as a subroutine for the twelve-split local to global map.

    i.e. Map local triangle vertex indices to their global variable indices
    NOTE: Also used as a subroutine to generate the twelve split maps

    @param[in] global_vertex_indices: global indices of the triangle vertices
    @param[in] num_variable_vertices: number of variable vertices
    @param[out] local_to_global_map: map from local to global DOF indices
    """

    dimension: int = 3
    # TODO: check if having -1 initialized into local_to_global_map is the way to go
    local_to_global_map: list[int] = [-1 for _ in range(27)]

    for local_vertex_index in range(3):
        global_vertex_index: int = global_vertex_indices[local_vertex_index]

        # Add vertex position index values
        for coord in range(dimension):
            local_index: int = generate_local_vertex_position_variable_index(
                local_vertex_index, coord, dimension)

            global_index: int
            if (global_vertex_index < 0):
                global_index = -1
            else:
                global_index = generate_global_vertex_position_variable_index(
                    global_vertex_index, coord, dimension)

            local_to_global_map[local_index] = global_index

        # Add vertex gradient index values
        for row in range(2):
            for col in range(dimension):
                local_index: int = generate_local_vertex_gradient_variable_index(
                    local_vertex_index, row, col, dimension)
                global_index: int
                if (global_vertex_index < 0):
                    global_index = -1
                else:
                    global_index = generate_global_vertex_gradient_variable_index(
                        num_variable_vertices, global_vertex_index, row, col, dimension)

                local_to_global_map[local_index] = global_index

    return local_to_global_map


def generate_twelve_split_local_to_global_map(global_vertex_indices: list[int],
                                              global_edge_indices: list[int],
                                              num_variable_vertices: int) -> list[int]:
    """
    Used in optimize_spline_surface.py

    Given the global vertex and edge indices of a triangle, compute the map
    from the local DOF vector indices for this triangle to their indices in
    the global DOF vector for the twelve-split.

    @param[in] global_vertex_indices: global indices of the triangle vertices
    @param[in] global_vertex_indices: global indices of the triangle edges 
    @param[in] num_variable_vertices: number of variable vertices
    @param[out] local_to_global_map: map from local to global DOF indices
    """
    # Get index map for the Powell-Sabin shared variables
    dimension: int = 3

    six_split_local_to_global_map: list[int] = generate_six_split_local_to_global_map(
        global_vertex_indices, num_variable_vertices)

    # TODO: should be returning a copy by value since both objects are list[int]
    local_to_global_map: list[int] = six_split_local_to_global_map.copy()
    assert len(local_to_global_map) == 36

    for local_edge_index in range(3):
        global_edge_index: int = global_edge_indices[local_edge_index]

        # Add edge gradient index values
        for coord in range(dimension):
            local_index: int = generate_local_edge_gradient_variable_index(
                local_edge_index, coord, dimension)
            global_index: int
            if (global_edge_index < 0):
                global_index = -1
            else:
                global_index = generate_global_edge_gradient_variable_index(
                    num_variable_vertices, global_edge_index, coord, dimension)

            local_to_global_map[local_index] = global_index

    return local_to_global_map


def update_independent_variable_vector(variable_values: VectorX, variable_vector: SpatialVector, start_index: int) -> SpatialVector:
    """
    Update variables in a vector from the vector of all variable values from some
    start index
    """
    # TODO: again, is it advised to edit a parameter and return said parameter?
    # TODO: could probably do this with NumPy indexing
    for i in range(variable_vector.size):
        variable_index: int = start_index + i
        # NOTE: SpatialVector is of shape (1,3)
        # TODO: is shape of VectorX still 1D like (n, )???
        variable_vector[0][i] = variable_values[variable_index]

    return variable_vector


def update_independent_variable_matrix(variable_values: VectorX, variable_matrix: Matrix2x3r, start_index: int) -> Matrix2x3r:
    """
    Update variables in a matrix from the vector of all variable values from some
    start index The flattening of the matrix is assumed to be row major.
    """
    dimension: int = variable_matrix.shape[1]  # Columns
    for i in range(variable_matrix.shape[0]):  # rows
        for j in range(variable_matrix.shape[1]):  # columns
            local_index: int = generate_local_variable_matrix_index(i, j, dimension)
            variable_index: int = start_index + local_index
            # TODO: is variable_values still 1D shape (n, )????
            variable_matrix[i, j] = variable_values[variable_index]

    return variable_matrix


def update_position_variables(variable_values: VectorX, variable_vertices: list[int]) -> list[SpatialVector]:
    """
    Used in optimize_spline_surface.py

    Extract vertex positions from the global DOF vector.
    i.e. Update all position variables

    @param[in] variable_values: twelve-split DOF vector
    @param[in] variable_vertices: list of variable vertex indices
    @param[out] vertex_positions: list of vertex position values
    """
    vertex_positions: list[SpatialVector] = []
    if (len(vertex_positions) == 0):
        # TODO: is it supposed to be such that it returns an empty vertex_position list?
        # Or does it modify vertex_positions as an argument and the returns it?
        return vertex_positions

    num_variable_vertices: int = len(variable_vertices)
    dimension: int = 3

    for vertex_index in range(num_variable_vertices):
        start_index: int = generate_global_vertex_position_variables_start_index(
            vertex_index, dimension)

        # FIXME: below accepts parameter, modifies it, then returns it. Is this the way to go?
        vertex_positions[variable_vertices[vertex_index]] = update_independent_variable_vector(
            variable_values, vertex_positions[variable_vertices[vertex_index]], start_index)

    return vertex_positions


def update_vertex_gradient_variables(variable_values: VectorX, variable_vertices: list[int], vertex_gradients: list[Matrix2x3r]) -> list[Matrix2x3r]:
    """
    Used in optimize_spline_surface.py

    Extract vertex gradients from the global DOF vector.
    i.e. Update all vertex gradient variables

    @param[in] variable_values: twelve-split DOF vector
    @param[in] variable_vertices: list of variable vertex indices
    @param[out] vertex_gradients: list of vertex gradient values
    """
    num_variable_vertices: int = len(variable_vertices)
    dimension: int = 3
    for vertex_index in range(num_variable_vertices):
        start_index: int = generate_global_vertex_gradient_variables_start_index(
            num_variable_vertices, vertex_index, dimension)

        vertex_gradients[variable_vertices[vertex_index]] = update_independent_variable_matrix(
            variable_values, vertex_gradients[variable_vertices[vertex_index]], start_index)

    return vertex_gradients


def update_edge_gradient_variables(variable_values: VectorX, variable_vertices: list[int], variable_edges: list[int], halfedge: Halfedge, he_to_corner: list[tuple[Index, Index]], edge_gradients: list[list[SpatialVector]]) -> list[list[SpatialVector]]:
    """
    Used in optimize_spline_surface.py

    Extract edge gradients from the global DOF vector.
    i.e. Update all edge gradient variables

    @param[in] variable_values: twelve-split DOF vector
    @param[in] variable_vertices: list of variable vertex indices
    @param[in] variable_edges: list of variable edge indices
    @param[in] halfedge: halfedge data structure
    @param[in] he_to_corner: map from halfedges to opposite triangle corners
    @param[out] edge_gradients: list of edge gradient values
    """
    # edge_gradients: list[list[SpatialVector]]

    dimension: int = 3
    num_variable_vertices: int = len(variable_vertices)
    num_variable_edges: int = len(variable_edges)

    # Get flat values for edge gradients
    for variable_edge_index in range(num_variable_edges):
        # Get corner for the given edge
        edge_index: int = variable_edges[variable_edge_index]
        first_halfedge_index: int = halfedge.edge_to_first_halfedge(edge_index)
        first_face_index: int = he_to_corner[first_halfedge_index][0]
        first_face_vertex_index: int = he_to_corner[first_halfedge_index][1]

        # Get index in the flattened variable vector
        start_index: int = generate_global_edge_gradient_variables_start_index(
            num_variable_vertices, variable_edge_index, dimension)

        # Update the gradients for the first corner
        edge_gradients[first_face_index][first_face_vertex_index] = update_independent_variable_vector(
            variable_values,
            edge_gradients[first_face_index][first_face_vertex_index],
            start_index)

        # Update the gradients for the second corner if it exists
        if not halfedge.is_boundary_edge(edge_index):
            second_halfedge_index: int = halfedge.edge_to_second_halfedge(edge_index)
            second_face_index: int = he_to_corner[second_halfedge_index][0]
            second_face_vertex_index: int = he_to_corner[second_halfedge_index][1]

            edge_gradients[second_face_index][second_face_vertex_index] = update_independent_variable_vector(
                variable_values,
                edge_gradients[second_face_index][second_face_vertex_index],
                start_index)

    return edge_gradients


def build_variable_vertex_indices_map(num_vertices: int, variable_vertices: list[int]):
    """
    Used in optimize_spline_surface.py

    Generate a map from all vertices to a list of variable vertices or -1 for
    vertices that are not variable.

    @param[in] num_vertices: total number of vertices
    @param[in] variable_vertices: list of variable vertex indices
    @param[out] global_vertex_indices: map from vertex indices to variable vertices
    """
    # Get variable vertex indices
    global_vertex_indices: list[int] = [-1 for _ in range(num_vertices)]
    for i in range(len(variable_vertices)):
        global_vertex_indices[variable_vertices[i]] = i

    return global_vertex_indices


def build_variable_edge_indices_map(num_faces: int, variable_edges: list[int], halfedge: Halfedge, he_to_corner: list[tuple[Index, Index]]) -> list[list[int]]:
    """
    Used in optimize_spline_surface.py

    Generate a map from all edges to a list of variable edges or -1 for
    edges that are not variable.
    @param[in] num_faces: total number of faces
    @param[in] variable_edges: list of variable edge indices
    @param[in] halfedge: halfedge data structure
    @param[in] he_to_corner: map from halfedges to opposite triangle corners
    @param[out] global_edge_indices: map from edge indices to variable edges
    """
    # TODO: does the below make a list of list[int, int, int] correctly?
    global_edge_indices: list[list[int]] = [[-1, -1, -1] for _ in range(num_faces)]

    for i in range(len(variable_edges)):
        edge_index = variable_edges[i]
        h0 = halfedge.edge_to_first_halfedge(edge_index)
        f0 = he_to_corner[h0][0]
        f0_vertex_index = he_to_corner[h0][1]
        global_edge_indices[f0][f0_vertex_index] = i

        if (not halfedge.is_boundary_edge(edge_index)):
            h1: int = halfedge.edge_to_second_halfedge(edge_index)
            f1: int = he_to_corner[h1][0]
            f1_vertex_index: int = he_to_corner[h1][1]
            global_edge_indices[f1][f1_vertex_index] = i

    return global_edge_indices


def update_energy_quadratic(local_energy: float,
                            local_derivatives: Gradient,
                            local_hessian: Hessian,
                            local_to_global_map: list[int],
                            num_independent_variables: int
                            # energy: float,
                            # derivatives: VectorX,
                            # hessian_entries: list[tuple[float, float, float]]
                            ) -> tuple[float, VectorX, list[tuple[float, float, float]]]:
    """
    Update global energy, derivatives, and hessian with local per face values

    @param[in] local_energy: local energy value
    @param[in] local_derivatives: local energy gradient
    @param[in] local_hessian: local energy Hessian
    @param[in] local_to_global_map: map from local to global DOF indices

    @param[out] energy: global energy value
    @param[out] derivatives: global energy gradient
    @param[out] hessian: global energy Hessian

    # TODO: is it right to pass in parameters that are then modified and then pass those same parameters back as return values?
    Returns: 
    energy: global energy value
    derivatives: global energy gradient
    hessian: global energy Hessian
    """
    energy: float
    derivatives: VectorX = np.zeros(shape=(num_independent_variables, ))
    hessian_entries: list[tuple[float, float, float]] = []
    logger.info("Adding local face energy %s", local_energy)
    logger.info("Local to global map: %s", local_to_global_map)

    # Update energy
    # Originally, energy was passed in, but I don't see another function that calls update_energy_qudaratic() that requires energy to be incremented like before.
    # Hence, energy just reassigned to local_energy.
    energy = local_energy
    # energy += local_energy

    # Update derivatives
    num_local_indices: int = len(local_to_global_map)
    for local_index in range(num_local_indices):
        global_index: int = local_to_global_map[local_index]
        if global_index < 0:
            continue  # Skip fixed variables with no global index
        derivatives[global_index] = local_derivatives[local_index]

    # Update hessian entries
    for local_index_i in range(num_local_indices):
        # Get global row index, skipping fixed variables with no global index
        global_index_i: int = local_to_global_map[local_index_i]
        if global_index_i < 0:
            continue

        for local_index_j in range(num_local_indices):
            # Get global column index, skipping fixed variables with no global index
            global_index_j: int = local_to_global_map[local_index_j]
            if global_index_j < 0:
                continue

            # Get Hessian entry value
            hessian_value: float = local_hessian[local_index_i, local_index_j]

            # Assemble global Hessian entry
            hessian_entries.append((global_index_i, global_index_j, hessian_value))

    return energy, derivatives, hessian_entries


def build_face_variable_vector(variables: list, i: int,
                               j: int,
                               k: int) -> list:
    """
    Build a triplet of face vertex values from a global array of vertex variables

    @param[in] variables: global variables
    @param[in] i: first variable index
    @param[in] j: second variable index
    @param[in] k: third variable index
    @param[out] face_variable_vector: variables for face Tijk
    """
    unimplemented("This function really is not needed in Python since arrays are already vectors.")
