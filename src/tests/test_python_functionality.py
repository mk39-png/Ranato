import os
import scipy as sp
import igl
import numpy as np
import mathutils


# #
# V_coeffs = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
# assert V_coeffs.shape == (3, 3)
# what = V_coeffs[[2], :]
# what = V_coeffs[[2]]
# print(what.shape)

# print(what)


# QQ_coeffs = np.zeros(shape=(5, 1))
# Qu_coeffs = np.zeros(shape=(5, 1))
# Qv_coeffs = np.zeros(shape=(5, 1))
# uv_coeffs = np.zeros(shape=(5, 1))
# uu_coeffs = np.zeros(shape=(5, 1))
# vv_coeffs = np.zeros(shape=(5, 1))

# monomial_coeffs = np.array([QQ_coeffs.flatten(),
#                             Qu_coeffs.flatten(),
#                             Qv_coeffs.flatten(),
#                             uv_coeffs.flatten(),
#                             uu_coeffs.flatten(),
#                             vv_coeffs.flatten()])


# print(monomial_coeffs.T.shape)

# # monomial_coeffs = np.zeros(shape=(5, 6))
# # monomial_coeffs[0:QQ_coeffs.size, 0] = QQ_coeffs
# # monomial_coeffs[0:Qu_coeffs.size, 1] = Qu_coeffs
# # monomial_coeffs[0:Qv_coeffs.size, 2] = Qv_coeffs
# # monomial_coeffs[0:uv_coeffs.size, 3] = uv_coeffs
# # monomial_coeffs[0:uu_coeffs.size, 4] = uu_coeffs
# # monomial_coeffs[0:vv_coeffs.size, 5] = vv_coeffs
# # print(monomial_coeffs)
# # print(monomial_coeffs.shape)

# num_faces = 4
# corner_to_edge = [[None, None, None] for _ in range(num_faces)]

# print(corner_to_edge)

# F = np.zeros(shape=[4, 3])
# print(F[0, :].shape)
# print(F[0].shape)
# print(F[:, 0].shape)

# mathutils.Quaternion
# vec = mathutils.Vector((1.0, 2.0, 3.0))
# vec = mathutils.Vector((1.0, 2.0, 3.0))

# https://blender.stackexchange.com/questions/159824/mathutils-matrix-matrix-world-set-get-round-trip-with-ndarray-requires-trans
# quat_b = mathutils.Quaternion(np.array([0.0, 1.0, 0.0])., math.radians(90.0))
# print(quat_b)
# X_WO = np.array([
#     [0, 1, 0],
#     [-1, 0, 0],
#     [0, 0, 1]])

# what = mathutils.Matrix(X_WO)
# print(what)

# huh = np.array(what)
# print(huh)

# yes = mathutils.Vector(np.array([[1, 2, 3]]))


# left = np.array([[1, 2, 3]])
# right = np.ones(shape=(2, 3))


# local_to_global_map: list[int] = [-1 for _ in range(27)]
# # print(local_to_global_map)

# num_faces = 2
# global_edge_indices: list[list[int]] = [[-1, -1, -1] for _ in range(num_faces)]
# print(global_edge_indices)
# https://stackoverflow.com/questions/8849833/python-list-reserving-space-resizing
# def list_resize(l: list, newsize: int, filling=None) -> None:
#     if newsize > len(l):
#         l.extend([filling for x in range(len(l), newsize)])
#     else:
#         del l[newsize:]


# corner_data = [[1] for _ in range(6)]
# sizing = 10

# list_resize(corner_data, sizing, [])
# print(corner_data)


# v = np.zeros(shape=(1, 2))
# f = np.zeros(shape=(1, 2))

# # F_submesh, V_submesh, _, _ = igl.remove_unreferenced(F, V)
# root_folder = os.getcwd()

# ret = igl.write_triangle_mesh(os.path.join(root_folder, "data", "bunny_out.obj"), v, f)

HASH_TABLE_SIZE = 2

hash_size_x: int = HASH_TABLE_SIZE
hash_size_y: int = HASH_TABLE_SIZE

# Clear the hash table
# NOTE: hash_table just going to be recreated in this method.
# NOTE: hash_table is HASH_TABLE_SIZE x HASH_TABLE_SIZE 2D list with elements list[int]
# hash_table: list[list[list[int]]] = [[[] for i in range(hash_size_x)] for j in range(hash_size_x)]

# print(len(hash_table))
# print(len(hash_table[0]))
# print(len(hash_table[0][0]))


# num_patches = 12
# num_boundaries = 3
# num_coeffs = 3
# patch_boundaries: list[list[np.ndarray]] = [
#     [np.zeros(shape=(num_coeffs, 1)) for _ in range(num_boundaries)]
#     for _ in range(num_patches)]
# assert len(patch_boundaries) == 12
# assert len(patch_boundaries[0]) == 3

# print(patch_boundaries)
# print(patch_boundaries[0])
# print(patch_boundaries[0][0].shape)
# # assert len(patch_boundaries[0][0].shape) == (3, 1)

# N = 3
# l: list[list[float]] = [[0.0 for _ in range(N)] for _ in range(N)]
# print(l)

# root_folder = os.getcwd()

V = np.array([
    [0., 0, 0],
    [1, 0, 0],
    [1, 1, 1],
    [2, 1, 0]
])

F = np.array([
    [0, 1, 2],
    [1, 3, 2]
])

print(igl.is_vertex_manifold(F))
# print(igl.pyigl_cor/e.is_border_vertex(F))
