
import numpy as np
import numpy.linalg as LA

from ..affine_manifold import *
Index = int


def remove_cones_from_uv(V: np.ndarray, F: np.ndarray, uv: np.ndarray, F_uv: np.ndarray, V_flat: np.ndarray, F_flat: np.ndarray, uv_flat: np.ndarray, F_uv_flat: np.ndarray) -> None:
    """
    Given a VF mesh with a global uv layout, remove the cones where the layout
    does not result in a flat metric.

    Args:
        V: input mesh vertices
        F: input mesh faces
        uv: input mesh uv coordinates
        F_uv: input mesh uv layout faces
        V_flat: mesh vertices with cones removed
        F_flat: mesh faces with cones removed
        uv_flat: mesh uv coordinates with cones removed
        F_uv_flat: mesh uv layout faces with cones removed

    Returns:
        None
    """
    # Double checking that these are indeed the types that they should be
    assert V.dtype == np.float64
    assert F.dtype == np.int64

    # Get the cones of the metric
    cone_manifold = AffineManifold(F, uv, F_uv)
    cones: list[Index] = cone_manifold.compute_cones()

    # TODO: do not use the version with coins...
    # logger.info("Removing cones at %s", formatted_vector(cones))
    logger.info("Removing cones at %s", cones)

    # Restrict the manifold to the flat vertices
    removed_faces: list[Index] = []
    remove_mesh_vertices(V, F, cones, V_flat, F_flat, removed_faces)

    # Remove the faces around cones from the uv layout as well
    remove_mesh_faces(uv, F_uv, removed_faces, uv_flat, F_uv_flat)


def generate_metric_from_uv(F: np.ndarray, uv: np.ndarray, l: list[list[float]]):
    # TODO: clarify the below is accessing rows and cols
    num_faces = F.shape[0]
    face_size = F.shape[1]

    assert face_size == 3
    l = [] * num_faces

    # Iterate over faces
    for i in range(num_faces):
        l[i] = [] * face_size

        # Iterate over vertices in face i
        for j in range(face_size):
            prev_uv = uv[F[i, (j + 2) % face_size], :]
            next_uv = uv[F[i, (j + 1) % face_size], :]
            edge_vector = prev_uv - next_uv

            # TODO: is this getting the right Norm?
            l[i][j] = LA.norm(edge_vector)
