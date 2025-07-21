from igl.pyigl_core import is_vertex_manifold, writeOBJ

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

    def __init__(self, patches: list[QuadraticSplineSurfacePatch]) -> None:
        self.clear()

        self.m_patches: list[QuadraticSplineSurfacePatch] = patches

        # Number of subdivisions per triangle of the domain
        self.num_subdivisions: int = 2

        # If true, compute unit length surface normal vectors
        self.normalize_surface_normals: bool = True


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

    def __init__(self, patches: list[QuadraticSplineSurfacePatch]):
        """
        Constructor from patches
        @param[in] patches: quadratic surface patches
        """
        # Protected
        self.m_patches: list[QuadraticSplineSurfacePatch] = patches

        # TODO: utilize some sort of pythonic hash table type
        #  Hash table data
        # hash_table is a 2D list of list[int]
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
        todo()
        points: list[SpatialVector]
        polylines: list[list[int]]

        for patch_index in range(self.num_patches):
            spline_surface_patch: QuadraticSplineSurfacePatch = self.get_patch(patch_index)
            # list of size 3
            patch_boundaries: list[LineSegment] = spline_surface_patch.get_domain.parametrize_patch_boundaries()

            for k, _ in enumerate(patch_boundaries):
                # Get points on the boundary curve
                list[PlanarPoint] parameter_points_k
                patch_boundaries[k].sample_points

        return points, polyline

    def save_obj(self, filename: str):
        """
        Save the triangulated surface as an obj.

        :param filename: filepath to save the obj
        :type filename: str
        """
        todo("Used in contour_network.py")

    def add_surface_to_viewer(self,
                              color: np.ndarray = SKY_BLUE,
                              num_subdivisions: int = DISCRETIZATION_LEVEL):
        """
        Add the surface to the viewer.

        :param color: color for the surface in the viewer
        :type color: np.ndarray

        :param num_subdivisions: number of subdivisions for the surface
        :type num_subdivisions: int
        """
        """
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        Eigen::MatrixXd N;

        # below is NOT pythonic way of intiailizing surfacE_disc_params... instEAD NEED to pass in num_subdivisions into it rather than setting it outside
        SurfaceDiscretizationParameters surface_disc_params;
        surface_disc_params.num_subdivisions = num_subdivisions;
        discretize(surface_disc_params, V, F, N);
        """
        todo("Used in twelve_split_spline.py and contour_network.py")

    def view(self, color: np.ndarray = SKY_BLUE, num_subdivisions: int = DISCRETIZATION_LEVEL):
        """
        View the surface.

        :param color: color for the surface in the viewer
        :type color: np.ndarray

        :param num_subdivisions: number of subdivisions for the surface
        :type num_subdivisions: np.ndarray

        :return: None
        """

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
        unimplemented()

    def deserialize(self):
        unimplemented()

    def write_spline(self):
        todo("Used in contour_network.py")

    def read_spline(self):
        unimplemented()

    def compute_patch_hash_tables(self):
        """
        Compute hash tables for the surface.
        """
        todo("Used in twelve_split_spline.py")

    def compute_hash_indices(self, point: PlanarPoint) -> tuple[int, int]:
        """
        Compute the hash indices of a point in the plane.

        :param point: point in the plane
        :return: pair
        """
        todo("Used in compute_ray_intersections.py")

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
        Compute bounding boxes for the patches
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
