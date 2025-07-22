from scipy.sparse import csr_matrix, coo_matrix
import os
import scipy as sp
import igl
import numpy as np
import mathutils
from cholespy import CholeskySolverD, MatrixType


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

# V = np.array([
#     [0., 0, 0],
#     [1, 0, 0],
#     [1, 1, 1],
#     [2, 1, 0]
# ])

# F = np.array([
#     [0, 1, 2],
#     [1, 3, 2]
# ])

# print(igl.is_vertex_manifold(F))
# # print(igl.pyigl_cor/e.is_border_vertex(F))

# r_alpha_flat = np.ones(shape=(36, 1))
# w_p = 2
# H_p = 2

# print((w_p * H_p) * np.dot(r_alpha_flat.T, r_alpha_flat)[0, 0])
# print(np.dot(r_alpha_flat.T, (w_p * H_p) * r_alpha_flat)[0, 0])


# alist = [(18, 53, 39), (42, 78, 51), (132, 38, 235)]

# # Expand each tuple (i, t, t) where i is the row index
# i, j, data = zip(*((i, t, t) for i, row in enumerate(alist) for t in row))
# print(i)
# print(j)


# # Build CSR matrix
# mat = csr_matrix((data, (i, j)), shape=(200, 150))

# print(mat.todense().shape)
# print(mat)

print(np.arange(20))
print(np.arange(20))
n_rows = 20
rows = np.arange(n_rows)
cols = np.arange(n_rows)
data = np.ones(n_rows)


hessian: csr_matrix = csr_matrix((data, (rows, cols)),
                                 shape=(n_rows, n_rows),
                                 dtype=float)
# hessian_coo = hessian.tocoo()

rows = hessian.indices
# cols = hessian.
# data = hessian.data
print(hessian.asformat("coo"))
num_rows = hessian.get_shape()[0]
# print()
# print(rows)
# print(cols)
# print(data)

solver = CholeskySolverD(num_rows - 1, rows, cols, data, MatrixType.CSR)

# hessian_entries = [(18.0, 53.0, 1), (42.2, 78.2, 1), (132, 38, 1)]
# indeX_rows, indeX_cols, values = zip(*hessian_entries)
# print(np.array(indeX_rows))

# # res = coo_matrix((values, (rows, cols)), shape=(133, 79)).tocsr()
# # reser = csr_matrix((values, (indeX_rows, indeX_cols)), shape=(133, 79), dtype=float)
# # resert: coo_matrix = reser.tocoo()
# resert = coo_matrix((values, (indeX_rows, indeX_cols)), shape=(133, 79), dtype=float)
# num_rows = n_rows
# print(resert.row)
# print(resert.col)
# print(resert.data)

# num_rows = resert.shape[0]
# rows = resert.row
# cols = resert.col
# data = resert.data

# print(resert.shape[0])  # rows
# print(np.array((1, 2, 3, 4)))
# solver = CholeskySolverD(n_rows, rows, cols, data, MatrixType.COO)
b = np.ones(n_rows, dtype=np.float64)
x = np.zeros_like(b, dtype=np.float64)

# NOTE: b in this case would be hessian
# Meanwhile x... well... that would be rhs!

print(solver.solve(b, x))
print(b)
print(x)


# print(reser)
