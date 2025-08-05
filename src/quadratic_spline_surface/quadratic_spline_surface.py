from igl.pyigl_core import is_vertex_manifold, writeOBJ
from polyscope.curve_network import CurveNetwork
from polyscope.surface_mesh import SurfaceMesh

from src.core.common import *
from src.core.evaluate_surface_normal import generate_quadratic_surface_normal_coeffs
from src.quadratic_spline_surface.position_data import *
from src.quadratic_spline_surface.quadratic_spline_surface_patch import *
from src.quadratic_spline_surface.quadratic_spline_surface_patch import QuadraticSplineSurfacePatch
from dataclasses import dataclass
from typing import TextIO
import os


@dataclass
class SurfaceDiscretizationParameters:
    """
    Parameters for the discretization of a quadratic spline
    """
    # Number of subdivisions per triangle of the domain
    num_subdivisions: int = 2
    # If true, compute unit length surface normal vectors
    normalize_surface_normals: bool = True


class QuadraticSplineSurface:
    """
    A piecewise quadratic surface.

    Supports:
    - evaluation
    - patch and subsurface extraction
    - triangulation
    - sampling
    - visualization
    - (basic) rendering
    - (de)serialization
    """
    # HACK: also has filename for deserializing patch info text file.
    # This is then used for testing

    def __init__(self, patches: list[QuadraticSplineSurfacePatch] | None = None, filename: str | None = None) -> None:
        """
        Constructor from patches
        @param[in] patches: quadratic surface patches
        """

        # HACK: adding initialization from file support
        # Sets m_patches to information from the input text file
        if filename is not None:
            self.read_spline(filename)
        else:
            # FIXME: m_patches may accidentally be set to None and mess up program
            # Protected
            self.m_patches: list[QuadraticSplineSurfacePatch] = patches

        # TODO: utilize some sort of pythonic hash table type
        #  Hash table data
        # hash_table is a 2D list of list[int]
        # NOTE: hash_table is HASH_TABLE_SIZE x HASH_TABLE_SIZE 2D list with elements list[int]
        # TODO: todo("Rename member variables below to show that they are part of the class...")
        # FIXME: I think the hash table is where everything goes wrong and is the one function I did not check yet...
        self.hash_table: list[list[list[int]]] = self.compute_patch_hash_tables()

        # TODO: what about the below? what is the reverse exactly?
        self.reverse_hash_table: list[list[tuple[int, int]]]

        # Hash table parameters
        self.patches_bbox_x_min: float = 0.0
        self.patches_bbox_x_max: float = 0.0
        self.patches_bbox_y_min: float = 0.0
        self.patches_bbox_y_max: float = 0.0
        self.hash_x_interval: float = 0.0
        self.hash_y_interval: float = 0.0

    @property
    def num_patches(self) -> PatchIndex:
        """
        Get the number of patches in the surface
        @return number of patches
        """
        return len(self.m_patches)

    def get_patch(self, patch_index: PatchIndex) -> QuadraticSplineSurfacePatch:
        """
        Get a reference to a spline patch
        @return spline patch
        """
        return self.m_patches[patch_index]

    def evaluate_patch(self, patch_index: PatchIndex, domain_point: PlanarPoint) -> SpatialVector:
        """
        Evaluate the surface at a given patch and domain point
        @param[in] patch_index: index of the patch to evaluate
        @param[in] domain_point: point in the patch domain to evaluate
        @param[out] surface_point: output point on the surface
        """
        surface_point: SpatialVector = self.get_patch(
            patch_index).evaluate(domain_point)
        assert surface_point.shape == (1, 3)
        return surface_point

    def evaluate_patch_normal(self, patch_index: PatchIndex, domain_point: PlanarPoint) -> SpatialVector:
        """
        Evaluate the surface normal at a given patch and domain point.

        :param patch_index: index of the patch to evaluate
        :type patch_index: PatchIndex
        :param domain_point: point in the patch domain to evaluate
        :type domain_point: PlanarPoint

        :return: output point on the surface
        :rtype: SpatialVector
        """

        surface_normal: SpatialVector = self.get_patch(
            patch_index).evaluate_normal(domain_point)
        assert surface_normal.shape == (1, 3)
        return surface_normal

    def empty(self) -> bool:
        """
        Determine if the surface is empty

        :return: true iff the surface is empty
        """
        return len(self.m_patches) == 0

    def clear(self) -> None:
        """
        Clear the surface
        """
        self.m_patches.clear()

    def subsurface(self, patch_indices: list[PatchIndex]) -> "QuadraticSplineSurface":
        """
        Generate a subsurface with the given patch indices.

        :param patch_indices: indices of the patches to keep.
        :type patch_indices: list[PatchIndex]
        :return: subsurface with the given patches
        :rtype: QuadraticSplineSurface
        """
        sub_patches: list[QuadraticSplineSurfacePatch] = []

        for i, _ in enumerate(patch_indices):
            sub_patches.append(self.m_patches[patch_indices[i]])

        subsurface_spline = QuadraticSplineSurface(sub_patches)
        return subsurface_spline

    def triangulate_patch(self,
                          patch_index: PatchIndex,
                          num_refinements: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Triangulate a given patch.

        :param patch_index: patch to triangulate
        :type patch_index: PatchIndex
        :param num_refinements: number of refinements for the triangulation
        :type num_refinements: int

        :return: vertices (V), faces (F), and vertex normals (N) of the triangulation
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        V: np.ndarray[tuple[int, int], np.dtype[np.float64]]
        F: np.ndarray[tuple[int, int], np.dtype[np.int64]]
        N: np.ndarray[tuple[int, int], np.dtype[np.float64]]
        V, F, N = self.get_patch(patch_index).triangulate(num_refinements)

        return V, F, N

    def discretize(self, surface_disc_params: SurfaceDiscretizationParameters
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Triangulate the surface.

        :param surface_disc_params: discretization parameters
        :type surface_disc_params: SurfaceDiscretizationParameters
        :return: vertices of the triangulation (V_tri), faces of the triangulation(F_tri), and vertex normals (N_tri)
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        num_subdivisions: int = surface_disc_params.num_subdivisions

        if self.empty():
            V: np.ndarray = np.ndarray(shape=(0, 0), dtype=np.float64)
            F: np.ndarray = np.ndarray(shape=(0, 0), dtype=np.int64)
            N: np.ndarray = np.ndarray(shape=(0, 0), dtype=np.float64)
            return V, F, N

        # ** Build triangulated surface as a copy **
        patch_index: PatchIndex = 0

        # Build patch 0 to get information like num_patch_vertices and num_patch_faces.
        V_patch_0: np.ndarray[tuple[int], np.dtype[np.float64]]
        F_patch_0: np.ndarray[tuple[int], np.dtype[np.int64]]
        N_patch_0: np.ndarray[tuple[int], np.dtype[np.float64]]
        V_patch_0, F_patch_0, N_patch_0 = self.triangulate_patch(patch_index, num_subdivisions)

        # if logger.isEnabledFor(logging.DEBUG):
        #     compare_eigen_numpy_matrix(
        #         "spot_control\\quadratic_spline_surface\\discretize\\F_triangulate_patch_0.csv", F_patch_0)
        #     compare_eigen_numpy_matrix(
        #         "spot_control\\quadratic_spline_surface\\discretize\\V_triangulate_patch_0.csv", V_patch_0)
        #     compare_eigen_numpy_matrix(
        #         "spot_control\\quadratic_spline_surface\\discretize\\N_triangulate_patch_0.csv", N_patch_0)

        num_patch_vertices: int = V_patch_0.shape[ROWS]
        num_patch_faces: int = F_patch_0.shape[ROWS]
        patch_index += 1

        # Set the patch 0 inside V, F, and N of the surface.
        V_tri: np.ndarray = np.zeros(shape=(num_patch_vertices * self.num_patches, 3), dtype=np.float64)
        F_tri: np.ndarray = np.zeros(shape=(num_patch_faces * self.num_patches, 3), dtype=np.int64)
        N_tri: np.ndarray = np.zeros(shape=(num_patch_vertices * self.num_patches, 3), dtype=np.float64)
        V_tri[:num_patch_vertices, :] = V_patch_0
        F_tri[:num_patch_faces, :] = F_patch_0
        N_tri[:num_patch_vertices, :] = N_patch_0

        # Building the rest of the patches. (e.g. the rest of the shape V, F, and N)
        # NOTE: expect V to be shape (235008, 3) for the Spot Control mesh
        while patch_index < self.num_patches:
            # NOTE: expect patches shape (24, 3) with num_subdivisions = 2 and the Spot Control mesh
            V_patch: np.ndarray[tuple[int, int], np.dtype[np.float64]]
            F_patch: np.ndarray[tuple[int, int], np.dtype[np.int64]]
            N_patch: np.ndarray[tuple[int, int], np.dtype[np.float64]]
            V_patch, F_patch, N_patch = self.triangulate_patch(patch_index, num_subdivisions)

            # FIXME: values stop being set after a certain point....
            V_tri[num_patch_vertices * patch_index: num_patch_vertices * (patch_index + 1),
                  : V_tri.shape[COLS]] = V_patch

            F_tri[num_patch_faces * patch_index: num_patch_faces * (patch_index + 1),
                  : F_tri.shape[COLS]] = F_patch + np.full(shape=(num_patch_faces, F_tri.shape[COLS]),
                                                           fill_value=num_patch_vertices * patch_index,
                                                           dtype=np.int64)
            N_tri[num_patch_vertices * patch_index: num_patch_vertices * (patch_index + 1),
                  : N_tri.shape[COLS]] = N_patch

            # XXX: need to increment patch_index
            patch_index += 1

        logger.info("%s surface vertices", V_tri.shape[ROWS])
        logger.info("%s surface faces", F_tri.shape[ROWS])
        logger.info("%s surface normals", N_tri.shape[ROWS])

        return V_tri, F_tri, N_tri

    def discretize_patch_boundaries(self) -> tuple[list[SpatialVector], list[list[int]]]:
        """
        Discretize all patch boundaries as polylines.
        NOTE: This also appears in contour_network folder in discretize.py, but is here for convenience and also for organization purposes.
        TODO: perhaps change to utilize NumPy arrays... but I'm a bit concerned about losing clarity. But I would gain proper shaping and would make everything clearer.

        :return points: list of polyline points.
        :rtype points: list[SpatialVector]

        :return polyline: list of lists of polyline edges
        :rtype polyline: list[list[int]]
        """
        points: list[SpatialVector] = []
        polylines: list[list[int]] = []

        # FIXME this part takes the longest. optimize please.
        for patch_index in range(self.num_patches):
            spline_surface_patch: QuadraticSplineSurfacePatch = self.get_patch(patch_index)
            # list of size 3
            patch_boundaries: list[LineSegment] = spline_surface_patch.get_domain.parametrize_patch_boundaries()

            for k, _ in enumerate(patch_boundaries):
                # Get points on the boundary curve
                parameter_points_k: list[PlanarPoint] = []
                patch_boundaries[k].sample_points(5, parameter_points_k)
                points_k: list[SpatialVector] = []

                for i, _ in enumerate(parameter_points_k):
                    points_k.append(spline_surface_patch.evaluate(parameter_points_k[i]))

                # Build polyline for the given curve
                polyline: list[int] = []
                for l, _ in enumerate(points_k):
                    polyline.append(len(points) + l)

                points.extend(points_k)
                polylines.append(polyline)

        return points, polylines

    def save_obj(self, filename: str) -> None:
        """
        Save the triangulated surface as an obj.

        NOTE: Used in contour_network.py

        :param filename: filepath to save the obj
        :type filename: str
        """
        # Generate mesh discretization
        V: np.ndarray
        # NOTE: TC and FTC intialization... is it equivalent to ASOC eigen code?
        TC: np.ndarray = np.ndarray(shape=(0, 0))
        F: np.ndarray
        FTC: np.ndarray = np.ndarray(shape=(0, 0))
        N: np.ndarray
        surface_disc_params: SurfaceDiscretizationParameters = SurfaceDiscretizationParameters()
        V, F, N = self.discretize(surface_disc_params)

        # Write mesh to file
        igl.writeOBJ(filename, V, F, N, F, TC, FTC)

    def add_surface_to_viewer(self,
                              color: Matrix3x1r = SKY_BLUE,
                              num_subdivisions: int = DISCRETIZATION_LEVEL) -> None:
        """
        Add the surface to the viewer.
        NOTE: Used in twelve_split_spline.py and contour_network.py

        :param color: color for the surface in the viewer
        :type color: np.ndarray

        :param num_subdivisions: number of subdivisions for the surface
        :type num_subdivisions: int
        """

        # TODO: adjust parameter naming of SurfaceDiscretizationParameters
        # Generate mesh discretization
        surface_disc_params = SurfaceDiscretizationParameters(num_subdivisions=num_subdivisions)
        V: np.ndarray[tuple[int, int], np.dtype[np.float64]]  # dtype float
        F: np.ndarray[tuple[int, int], np.dtype[np.int64]]  # dtype int
        N: np.ndarray[tuple[int, int], np.dtype[np.float64]]  # dtype float
        V, F, N = self.discretize(surface_disc_params)  # NOTE: this takes approx 10 sec to do
        # FIXME maybe discretization is wrong...

        # Add surface mesh
        ps.init()
        surface: SurfaceMesh = ps.register_surface_mesh("surface", V, F)
        surface.set_edge_width(0)
        surface.set_color(color.flatten())

        # Discretize patch boundaries
        boundary_points: list[SpatialVector]
        boundary_polylines: list[list[int]]

        # FIXME: number of boundary_polylines is wrong...
        boundary_points, boundary_polylines = self.discretize_patch_boundaries()

        # View contour curve network
        boundary_points_matrix: np.ndarray = convert_nested_vector_to_matrix(boundary_points)
        boundary_edges: list[list[int]] = convert_polylines_to_edges(boundary_polylines)

        # HACK: converting boundary edges to numpy array so that polyscope works, but may
        # want to have convert_polylines_to_edges return a Nx2 matrix by default, where each row is an edge.
        # FIXME is the below taking a bunch of time to do?
        patch_boundaries: CurveNetwork = ps.register_curve_network("patch_boundaries",
                                                                   boundary_points_matrix,
                                                                   np.array(boundary_edges))
        patch_boundaries.set_color((0.670, 0.673, 0.292))
        patch_boundaries.set_radius(0.0005)
        patch_boundaries.set_radius(0.0005)
        patch_boundaries.set_enabled(False)

        # if logger.level == logging.DEBUG:
        #     compare_eigen_numpy_matrix(
        #         "spot_control\\quadratic_spline_surface\\add_surface_to_viewer\\F_discretized_2_subdiv.csv", F)
        #     compare_eigen_numpy_matrix(
        #         "spot_control\\quadratic_spline_surface\\add_surface_to_viewer\\V_discretized_2_subdiv.csv", V)
        #     compare_eigen_numpy_matrix(
        #         "spot_control\\quadratic_spline_surface\\add_surface_to_viewer\\N_discretized_2_subdiv.csv", N)
        #     compare_eigen_numpy_matrix("spot_control\\quadratic_spline_surface\\add_surface_to_viewer\\boundary_points.csv",
        #                                np.array(boundary_points).squeeze())
        #     compare_eigen_numpy_matrix("spot_control\\quadratic_spline_surface\\add_surface_to_viewer\\boundary_polylines.csv",
        #                                np.array(boundary_polylines))
        #     compare_eigen_numpy_matrix("spot_control\\quadratic_spline_surface\\add_surface_to_viewer\\boundary_points_mat.csv",
        #                                np.array(boundary_points_matrix))
        #     compare_eigen_numpy_matrix("spot_control\\quadratic_spline_surface\\add_surface_to_viewer\\boundary_edges.csv",
        #                                np.array(boundary_edges).squeeze())

    def view(self, color: Matrix3x1r = SKY_BLUE, num_subdivisions: int = DISCRETIZATION_LEVEL) -> None:
        """
        View the surface.

        :param color: color for the surface in the viewer
        :type color: np.ndarray

        :param num_subdivisions: number of subdivisions for the surface
        :type num_subdivisions: np.ndarray

        :return: None
        """
        self.add_surface_to_viewer(color, num_subdivisions)
        ps.show()

    def screenshot(self,
                   filename: str,
                   camera_position: SpatialVector = np.array([[0.0, 0.0, 2.0]], dtype=np.float64),
                   camera_target: SpatialVector = np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
                   use_orthographic: bool = False) -> None:
        """
        Save a screenshot of the surface in the viewer.

        :param filename: file to save the screenshot.
        :type filename: str
        :param camera_position: camera position for the screenshot. (np.ndarray shape (1, 3))
        :type camera_position: SpatialVector
        :param camera_target: camera target for the screenshot. (np.ndarray shape (1, 3))
        :type camera_target: SpatialVector
        :param use_orthographic: use orthographic perspective if true.
        :type use_orthographic: bool

        :return: None
        """
        self.add_surface_to_viewer()

        ps.look_at(camera_position, camera_target)
        if use_orthographic:
            ps.set_view_projection_mode("orthographic")
        else:
            ps.set_view_projection_mode("perspective")
        ps.screenshot(filename)
        logger.info("Screenshot saved to %s", filename)
        ps.remove_all_structures()

    def serialize(self, output_file: TextIO) -> None:
        """
        Serialize the surface
        NOTE: used by write_spline()

        patch information in the format
            patch
            cx a_0 a_u a_v a_uv a_uu a_vv
            cy a_1 a_u a_v a_uv a_uu a_vv
            cz a_2 a_u a_v a_uv a_uu a_vv
            p1 p1_u p1_v
            p2 p2_u p2_v
            p3 p3_u p3_v

        @param[in] out: output stream for the surface
        """
        for i, _ in enumerate(self.m_patches):
            self.m_patches[i].serialize(output_file)

    def deserialize(self, input_file: TextIO) -> list[QuadraticSplineSurfacePatch]:
        """
        Deserialize a surface

        NOTE: used for testing with original ASOC code
        TODO: future implementations could port over to JSON for univeral formatting
        and better parsing

        patch information in the format
            patch
            cx a_0 a_u a_v a_uv a_uu a_vv
            cy a_1 a_u a_v a_uv a_uu a_vv
            cz a_2 a_u a_v a_uv a_uu a_vv
            p1 p1_u p1_v
            p2 p2_u p2_v
            p3 p3_u p3_v

        @param[in] in: input stream for the surface
        """
        # self.m_patches.clear()
        patches: list[QuadraticSplineSurfacePatch] = []

        patch_info_lines: list[str] = input_file.readlines()
        EXTRA_NEWLINE = 1
        ROWS_OF_PATCH_INFORMATION = 7
        NUM_OF_ROWS: int = len(patch_info_lines)
        assert NUM_OF_ROWS % ROWS_OF_PATCH_INFORMATION == 0

        # -- Read coordinate coefficients cx, cy, and cz along with point information --
        # NOTE: this relies on there being 7 rows for patch format
        for i in range(0, NUM_OF_ROWS, ROWS_OF_PATCH_INFORMATION):
            # TODO: add better checking and optional comments ettter

            # TODO: use regex to verify cx, cy, cz pattern
            # Read coordinates (skipping the label and reading the float data)
            cx = np.array(list(map(float, patch_info_lines[i + 1].split()[1:])), dtype=np.float64)
            cy = np.array(list(map(float, patch_info_lines[i + 2].split()[1:])), dtype=np.float64)
            cz = np.array(list(map(float, patch_info_lines[i + 3].split()[1:])), dtype=np.float64)
            surface_mapping_coeffs: Matrix6x3r = np.stack((cx, cy, cz), axis=1, dtype=np.float64)
            assert surface_mapping_coeffs.shape == (6, 3)

            # TODO: use regex to verify p1, p2, p3 pattern
            p1 = np.array(list(map(float, patch_info_lines[i + 4].split()[1:])), dtype=np.float64)
            p2 = np.array(list(map(float, patch_info_lines[i + 5].split()[1:])), dtype=np.float64)
            p3 = np.array(list(map(float, patch_info_lines[i + 6].split()[1:])), dtype=np.float64)
            vertices: Matrix3x2r = np.array([p1, p2, p3], dtype=np.float64)
            assert vertices.shape == (3, 2)

            domain: ConvexPolygon = ConvexPolygon.init_from_vertices(vertices)

            # Add patch to the spline surface
            patches.append(QuadraticSplineSurfacePatch(surface_mapping_coeffs, domain))

        return patches

    # TODO: this should be accessible OUTSIDE the class
    def write_spline(self, filename: str) -> None:
        """
        Write the surface serialization to file.
        NOTE: used in contour_network.py

        patch information in the format
            patch
            cx a_0 a_u a_v a_uv a_uu a_vv
            cy a_1 a_u a_v a_uv a_uu a_vv
            cz a_2 a_u a_v a_uv a_uu a_vv
            p1 p1_u p1_v
            p2 p2_u p2_v
            p3 p3_u p3_v

        @param[in] filename: file path for the serialized surface
        """
        logger.info("Writing spline to %s", filename)

        filepath: str = os.path.abspath(f"src\\tests\\{filename}")
        # TODO: check if the file exists before writing to it....

        if os.path.isfile(filepath):
            logger.warning("Overwritting spline txt file at %s", filepath)
            # raise Exception("File already exists. Choose different file name to write spline to.")

        with open(filepath, 'w', encoding='utf-8') as output_file:
            self.serialize(output_file)
        output_file.close()

    # This should be accessible OUTISDE the class to then create a new QuadraticSplineSurface object for debugging.
    def read_spline(self, filename: str) -> None:
        """
        Read a surface serialization from file
        @param[in] filename: file path for the serialized surface
        @param[out] self.m_patches: patches to save to.

        NOTE: method used for testing with ASOC code and to make sure that implementation is correct.
        """
        input_file: TextIO
        filepath: str = os.path.abspath(f"src\\tests\\{filename}")
        if not os.path.isfile(filepath):
            raise Exception("File does not exist. Choose a file to read spline from.")

        with open(filepath, 'r', encoding='utf-8') as input_file:
            self.m_patches = self.deserialize(input_file)
        input_file.close()

    def compute_patch_hash_tables(self) -> list[list[list[int]]]:
        """
        Compute hash tables for the surface.
        NOTE: Used in twelve_split_spline.py
        """
        num_patch: int = self.num_patches
        hash_size_x: int = HASH_TABLE_SIZE
        hash_size_y: int = HASH_TABLE_SIZE

        # Clear the hash table
        # NOTE: hash_table just going to be recreated in this method.
        # NOTE: hash_table is HASH_TABLE_SIZE x HASH_TABLE_SIZE 2D list with elements list[int]
        hash_table: list[list[list[int]]] = [
            [[] for _ in range(hash_size_x)]
            for _ in range(hash_size_x)
        ]

        # Compute bounding box for all the patches
        self.__compute_patches_bbox()
        x_min: float = self.patches_bbox_x_min
        x_max: float = self.patches_bbox_x_max
        y_min: float = self.patches_bbox_y_min
        y_max: float = self.patches_bbox_y_max

        for i in range(1, num_patch):
            if (x_min > self.m_patches[i].get_bbox_x_min()):
                x_min = self.m_patches[i].get_bbox_x_min()
            if (x_max < self.m_patches[i].get_bbox_x_max()):
                x_max = self.m_patches[i].get_bbox_x_max()
            if (y_min > self.m_patches[i].get_bbox_y_min()):
                y_min = self.m_patches[i].get_bbox_y_min()
            if (y_max < self.m_patches[i].get_bbox_y_max()):
                y_max = self.m_patches[i].get_bbox_y_max()

        x_interval: float = (x_max - x_min) / hash_size_x
        y_interval: float = (y_max - y_min) / hash_size_y

        self.hash_x_interval = x_interval
        self.hash_y_interval = y_interval

        eps: float = 1e-10

        # Hash into each box
        # FIXME below takes 2 minutes to run...
        # FIXME below sometimes has NaN inside m_patches, other times no. Like first time running has NaN
        for i in range(num_patch):
            left_x: int = int((self.m_patches[i].get_bbox_x_min() - eps - x_min) / x_interval)
            right_x: int = int(hash_size_x - int((x_max - self.m_patches[i].get_bbox_x_max() - eps) / x_interval) - 1)
            left_y: int = int((self.m_patches[i].get_bbox_y_min() - eps - y_min) / y_interval)
            right_y: int = int(hash_size_y - int((y_max - self.m_patches[i].get_bbox_y_max() - eps) / y_interval) - 1)

            for j in range(left_x, right_x + 1):
                for k in range(left_y, right_y + 1):
                    hash_table[j][k].append(i)

        return hash_table

    def compute_hash_indices(self, point: PlanarPoint) -> tuple[int, int]:
        """
        Compute the hash indices of a point in the plane.
        NOTE: Used in compute_ray_intersections.py

        :param point: PlanarPoint object of shape (1, 2) to convert to hash table x and y values
        :type point: PlanarPoint

        :return: tuple of hash_x and hash_y computed.
        """
        hash_x = int((point[0][0] - self.patches_bbox_x_min) / self.hash_x_interval)
        hash_y = int((point[0][1] - self.patches_bbox_y_min) / self.hash_y_interval)

        if (hash_x < 0) or (hash_x >= HASH_TABLE_SIZE):
            logger.error("x hash index out of bounds")
            hash_x: int = max(min(hash_x, HASH_TABLE_SIZE - 1), 0)

        if (hash_y < 0) or (hash_y >= HASH_TABLE_SIZE):
            logger.error("y hash index out of bounds")
            hash_y: int = max(min(hash_y, HASH_TABLE_SIZE - 1), 0)

        return (hash_x, hash_y)

    # ***************
    # Private Methods
    # ***************

    def __is_valid_patch_index(self, patch_index: PatchIndex) -> bool:
        """
        Determine if a patch index is valid
        """
        if patch_index >= self.num_patches:
            return False

        return True

    def __compute_patches_bbox(self) -> None:
        """
        Compute bounding boxes for the patches.
        As in, calculates values for member variables below:
        - self.patches_bbox_x_min
        - self.patches_bbox_x_max
        - self.patches_bbox_y_min
        - self.patches_bbox_y_max
        """
        x_min: float = self.m_patches[0].get_bbox_x_min()
        x_max: float = self.m_patches[0].get_bbox_x_max()
        y_min: float = self.m_patches[0].get_bbox_y_min()
        y_max: float = self.m_patches[0].get_bbox_y_max()

        # FIXME: Why was it all NaN and now it's all fine???
        for i in range(1, self.num_patches):
            if (x_min > self.m_patches[i].get_bbox_x_min()):
                x_min = self.m_patches[i].get_bbox_x_min()
            if (x_max < self.m_patches[i].get_bbox_x_max()):
                x_max = self.m_patches[i].get_bbox_x_max()
            if (y_min > self.m_patches[i].get_bbox_y_min()):
                y_min = self.m_patches[i].get_bbox_y_min()
            if (y_max < self.m_patches[i].get_bbox_y_max()):
                y_max = self.m_patches[i].get_bbox_y_max()

        # TODO: isn't it more pythonic to return these in a tuple?
        self.patches_bbox_x_min: float = x_min
        self.patches_bbox_x_max: float = x_max
        self.patches_bbox_y_min: float = y_min
        self.patches_bbox_y_max: float = y_max
