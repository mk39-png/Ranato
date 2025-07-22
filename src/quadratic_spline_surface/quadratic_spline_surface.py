from igl.pyigl_core import is_vertex_manifold, writeOBJ
from polyscope.surface_mesh import SurfaceMesh

from src.core.common import *
from src.core.evaluate_surface_normal import generate_quadratic_surface_normal_coeffs
from src.quadratic_spline_surface.position_data import *
from src.quadratic_spline_surface.quadratic_spline_surface_patch import *
from src.quadratic_spline_surface.quadratic_spline_surface_patch import QuadraticSplineSurfacePatch

PatchIndex = int


class SurfaceDiscretizationParameters:
    """
    Parameters for the discretization of a quadratic spline
    """

    def __init__(self,
                 #  TODO: adjust naming of parameters
                 num_subdivisions: int = 2,
                 normalize_surface_normals: bool = True) -> None:
        # self.clear()
        # self.m_patches: list[QuadraticSplineSurfacePatch] = patches

        # Number of subdivisions per triangle of the domain
        self.num_subdivisions: int = num_subdivisions

        # If true, compute unit length surface normal vectors
        self.normalize_surface_normals: bool = normalize_surface_normals


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

    def __init__(self, patches: list[QuadraticSplineSurfacePatch]) -> None:
        """
        Constructor from patches
        @param[in] patches: quadratic surface patches
        """
        # Protected
        self.m_patches: list[QuadraticSplineSurfacePatch] = patches

        # TODO: utilize some sort of pythonic hash table type
        #  Hash table data
        # hash_table is a 2D list of list[int]
        # NOTE: hash_table is HASH_TABLE_SIZE x HASH_TABLE_SIZE 2D list with elements list[int]
        todo("Rename member variables below to show that they are part of the class...")
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

    def evaluate_patch(self, patch_index: PatchIndex, domain_point: PlanarPoint):
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

    def triangulate_patch(self, patch_index: PatchIndex, num_refinements: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Triangulate a given patch.

        :param patch_index: patch to triangulate
        :type patch_index: PatchIndex
        :param num_refinements: number of refinements for the triangulation
        :type num_refinements: int

        :return: vertices (V), faces (F), and vertex normals (N) of the triangulation
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """

        todo("Parameter modification and return below. So, change V F and N and return them properly according to Python")
        V: np.ndarray = np.zeros(shape=(0, 0))
        F: np.ndarray = np.zeros(shape=(0, 0))
        N: np.ndarray = np.zeros(shape=(0, 0))
        self.get_patch(patch_index).triangulate(num_refinements, V, F, N)

        return V, F, N

    def discretize(self, surface_disc_params: SurfaceDiscretizationParameters) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Triangulate the surface.

        :param surface_disc_params: discretization parameters
        :type surface_disc_params: SurfaceDiscretizationParameters

        :param V: vertices of the triangulation
        :type V: np.ndarray

        :param F: faces of the triangulation
        :type F: np.ndarray

        :param N: vertex normals
        :type N: np.ndarray

        :return: vertices of the triangulation (V_tri), faces of the triangulation(F_tri), and vertex normals (N_tri)
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        num_subdivisions: int = surface_disc_params.num_subdivisions

        if (self.empty()):
            # TODO: adjust return value here...
            return

        # Build triangulated surface in place
        patch_index: PatchIndex = 0
        V: np.ndarray  # dtype float
        F: np.ndarray  # dtype int
        N: np.ndarray  # dtype float
        V, F, N = self.triangulate_patch(patch_index, num_subdivisions)
        num_patch_vertices: int = V.shape[0]  # rows
        num_patch_faces: int = F.shape[0]  # rows
        patch_index += 1

        for _ in range(self.num_patches):
            V_patch: np.ndarray  # dtype float
            F_patch: np.ndarray  # dtype int
            N_patch: np.ndarray  # dtype float
            V_patch, F_patch, N_patch = self.triangulate_patch(patch_index, num_subdivisions)

            # TODO: double check dimensionality... in fact... add dimensions to types because this is all getting quite confusing.
            V[num_patch_vertices * patch_index: num_patch_vertices * (patch_index + 1),
              0: V.shape[1]] = V_patch
            F[num_patch_faces * patch_index: num_patch_faces * (patch_index + 1),
              0: F.shape[1]] = F_patch + np.full(shape=(num_patch_faces, F.shape[1]), fill_value=num_patch_vertices * patch_index, dtype=int)
            N[num_patch_vertices * patch_index: num_patch_vertices * (patch_index + 1),
              0: N.shape[1]] = N_patch

        logger.info("%s surface vertices", V.shape[0])
        logger.info("%s surface faces", F.shape[0])
        logger.info("%s surface normals", N.shape[0])

        return V, F, N

    def discretize_patch_boundaries(self) -> tuple[list[SpatialVector], list[list[int]]]:
        """
        Discretize all patch boundaries as polylines.
        NOTE: This also appears in contour_network folder in discretize.py, but is here for convenience and also for organiztion purposes.

        :return points: list of polyline points.
        :rtype points: list[SpatialVector]

        :return polyline: list of lists of polyline edges
        :rtype polyline: list[list[int]]
        """
        points: list[SpatialVector] = []
        polylines: list[list[int]] = []

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

    def save_obj(self, filename: str):
        """
        Save the triangulated surface as an obj.

        :param filename: filepath to save the obj
        :type filename: str
        """
        todo("Used in contour_network.py")
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
                              color: np.ndarray = SKY_BLUE,
                              num_subdivisions: int = DISCRETIZATION_LEVEL):
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
        V: np.ndarray  # dtype float
        F: np.ndarray  # dtype int
        N: np.ndarray  # dtype float
        V, F, N = self.discretize(surface_disc_params)

        # Add surface mesh
        ps.init()
        surface: SurfaceMesh = ps.register_surface_mesh("surface", V, F)
        surface.set_edge_width(0)
        surface.set_color(color)

        # Discretize patch boundaries
        boundary_points: list[SpatialVector]
        boundary_poly_lines: list[list[int]]
        boundary_points, boundary_poly_lines = self.discretize_patch_boundaries()

        # View contour curve network
        # boundary_points_mat = convert_nested_vector_to_matrix
        todo()

    def view(self, color: np.ndarray = SKY_BLUE, num_subdivisions: int = DISCRETIZATION_LEVEL):
        """
        View the surface.

        :param color: color for the surface in the viewer
        :type color: np.ndarray

        :param num_subdivisions: number of subdivisions for the surface
        :type num_subdivisions: np.ndarray

        :return: None
        """
        todo()

    def screenshot(self, filename: str,
                   camera_postion: SpatialVector = np.array(
                       [[0.], [0.], [2.]]),
                   camera_target=np.array([[0.], [0.], [0.]]),
                   use_orthographic: bool = False) -> None:
        # TODO: include types in docstring
        """
        Save a screenshot of the surface in the viewer.

        :param filename: file to save the screenshot.
        :param camera_position: camera position for the screenshot.
        :param camera_target: camera target for the screenshot.
        :param use_orthographic: use orthographic perspective if true.
        """

    def serialize(self):
        """
        Serialize the surface
        @param[in] out: output stream for the surface
        """
        unimplemented()

    def deserialize(self):
        """
        Deserialize a surface
        @param[in] in: input stream for the surface
        """
        unimplemented()

    def write_spline(self):
        """
        Write the surface serialization to file
        @param[in] filename: file path for the serialized surface
        """
        todo("Used in contour_network.py")

    def read_spline(self):
        """
        Read a surface serialization from file
        @param[in] filename: file path for the serialized surface
        """
        unimplemented()

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

    def __is_valid_patch_index(self, patch_index: PatchIndex):
        """
        Determine if a patch index is valid
        """
        if patch_index >= self.num_patches:
            return False

        return True

    def __compute_patches_bbox(self):
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
