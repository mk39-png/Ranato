
import numpy as np
from cholespy import CholeskySolverD, MatrixType
from scipy.sparse import coo_matrix

# from ranato.algebraic_contours.contour_network.intersection_data import IntersectionData
# from ranato.algebraic_contours.core.common import Matrix4x4f

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

n_rows = 30000000


def make_hessian_inverse():

    # TODO: below works fine for really large matrix....
    # So maybe not a problem with the matrix size and datatype, but rather
    print(np.arange(20))
    print(np.arange(20))
    rows = np.arange(n_rows)
    cols = np.arange(n_rows)
    data = np.ones(n_rows)

    hessian: coo_matrix = coo_matrix((data, (rows, cols)),
                                     shape=(n_rows, n_rows),
                                     dtype=float)

    hessian2: coo_matrix = coo_matrix((data, (rows, cols)),
                                      shape=(n_rows, n_rows),
                                      dtype=float)
    hessian3: coo_matrix = coo_matrix((data, (rows, cols)),
                                      shape=(n_rows, n_rows),
                                      dtype=float)
    hessian4: coo_matrix = coo_matrix((data, (rows, cols)),
                                      shape=(n_rows, n_rows),
                                      dtype=float)
    hessian5: coo_matrix = coo_matrix((data, (rows, cols)),
                                      shape=(n_rows, n_rows),
                                      dtype=float)
    hessian6: coo_matrix = coo_matrix((data, (rows, cols)),
                                      shape=(n_rows, n_rows),
                                      dtype=float)
    # hessian_coo = hessian.tocoo()

    # rows = hessian.indices
    # cols = hessian.
    data = hessian.data
    print(hessian.asformat("coo"))
    num_rows = hessian.get_shape()[0]
    # print()
    # print(rows)
    # print(cols)
    # print(data)

    solver = CholeskySolverD(num_rows - 1, rows, cols, data, MatrixType.COO)

    return solver


def main():
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

    # solver = make_hessian_inverse()

    # print(solver.solve(b, x))
    # print(b)
    # print(x)

    # print(reser)

    # six_split_local_to_global_map: list[int] = [-1 for _ in range(27)]
    # local_to_global_map: list[int] = [39 for _ in range(36)]
    # local_to_global_map[0:len(six_split_local_to_global_map)] = six_split_local_to_global_map

    # print(local_to_global_map)

    # Seeing if array broadcasting worked... found out that shape of V was not working as intended.
    testing = np.ones(shape=(24, 3))
    num_patch_vertices = 24
    patch_index = 1
    # V[num_patch_vertices * patch_index: num_patch_vertices * (patch_index + 1),
    #   0: V.shape[1]]

    # num_patches = 10
    # patch_boundary_contour_map: list[list[int]] = [[-1, -1, -1] for _ in range(num_patches)]
    # patch_boundary_contour_map[0][0] = 39
    # print(patch_boundary_contour_map)

    # is_boundary_patch: list[bool] = [False] * num_patches
    # is_boundary_patch[0] = True
    # print(is_boundary_patch)

    # print(np.array([1]).flatten())

    # a = np.array([0, 0, 0, 1, 1, 1])
    # b = np.trim_zeros(np.copy(a))
    # b[0] = 39
    # print(a)
    # print(b)

    # block = np.arange(1, 17)
    # block.resize((4, 4))
    # print(block)
    # print(block[0:2, 0:3])

    # test = np.ndarray(shape=(2, 3))
    # print(test)
    # print(test[:, 0])
    # print(test[0:1, 0:2])
    # print(np.arange(3))
    # print(np.arange(3) @ np.arange(3))
    # vector_line: list[Number] = [np.float64(10), np.float64(10),
    #                              np.float64(50), np.float64(50),
    #                              np.float64(100), np.float64(120),]
    # point_2d = [40, 100]
    # point_2d_2 = [0, 43]
    # color = (1, 1, 255, 1)
    # svg_elements = [
    #     svg.Polyline(points=vector_line,
    #                  stroke="black",
    #                  fill="transparent",
    #                  stroke_width=1.0),
    #     svg.Circle(cx=point_2d[0], cy=point_2d[1], r=10, fill=f"rgba{color}"),
    #     svg.Circle(cx=point_2d_2[0], cy=point_2d_2[1], r=10, fill=f"rgba{color}")
    # ]

    # # svg_elements.append(svg.Circle(cx=point_2d[0], cy=point_2d[1], r=10, fill=f"rgba{color}"))
    # # svg_elements.append(svg.Circle(cx=point_2d[0], cy=point_2d[1], r=10, fill=f"rgba{color}"))
    # test_svg = svg.SVG(x=0, y=0, width=120, height=120, elements=svg_elements)
    # filename = "svg_testing.svg"
    # filepath = os.path.abspath(f"src\\tests\\{filename}")
    # print(test_svg.as_str())

    # with open(filepath, 'w', encoding='utf-8') as output_file:
    #     output_file.write(test_svg.as_str())

    # lister = []
    # lister.extend((1, 2, 3))
    # print(lister)

    # print(svg.SVG(
    #     # x=0, y=0,
    #     viewBox=svg.ViewBoxSpec(0, 0, 120, 120),
    #     # width=120, height=120,
    #     elements=[svg.Polygon(
    #         points=[60, 30, 90, 90, 30, 90],
    #         elements=[svg.AnimateTransform(
    #             attributeName="transform",
    #             type="rotate",
    #             from_="0 60 70",
    #             to="360 60 70",
    #             dur=timedelta(seconds=10),
    #             repeatCount="indefinite",
    #         )]
    #     )]
    # ))

    # # Initialize matrix to the identity
    # translation = np.array([3, 4, 5])
    # translation_matrix: Matrix4x4f = np.identity(4, dtype=np.float64)
    # assert translation_matrix.shape == (4, 4)

    # # Add translation using homogeneous coordinates
    # print(translation_matrix)
    # translation_matrix[:3, 3:4] = translation.reshape((3, 1))
    # print(translation_matrix)

    frame = np.arange(0, 9)
    frame.resize((3, 3))
    print(frame.T)

    rotation_matrix: Matrix4x4f = np.zeros(shape=(4, 4), dtype=np.float64)
    print(rotation_matrix)

    # The desired rotation is the transpose of the frame
    rotation_matrix[0:3, 0:3] = frame.T

    # No homoegeneous scaling for the rotation
    rotation_matrix[3, 3] = 1
    print(rotation_matrix)

    point = np.array([2, 3, 4])
    homogeneous_coords: Vector4f = np.ones(shape=(4, ), dtype=np.float64)
    homogeneous_coords[:3] = point
    print(homogeneous_coords)


# def test_segment():
#     segment_intersection_data: list[IntersectionData] = []
#     print()
#     for i in range(10):
#         segment_intersection_data.append(IntersectionData(i, i, i, i))

#     for i in range(10):
#         # segment_intersection_data.append(IntersectionData(i, i, i, i))
#         print(segment_intersection_data[i].knot)

#     segment_intersection_data.sort(key=lambda data: data.knot)

#     print("----------------")
#     for i in range(10):
#         # segment_intersection_data.append(IntersectionData(i, i, i, i))
#         print(segment_intersection_data[i].knot)
if __name__ == "__main__":
    main()
