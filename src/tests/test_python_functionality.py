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


left = np.array([[1, 2, 3]])
right = np.ones(shape=(2, 3))
