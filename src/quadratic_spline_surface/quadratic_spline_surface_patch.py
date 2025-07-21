"""
Representation for quadratic surface patches with convex domains.
"""
from src.core.common import *
from src.core.convex_polygon import *
from src.core.convex_polygon import ConvexPolygon
from src.core.evaluate_surface_normal import *
from src.core.polynomial_function import *
from src.core.rational_function import *
from src.core.bivariate_quadratic_function import *

import polyscope as ps
# TODO: igl is_vertex_manifold... or maybe mathutils?


# **************************************
# Quadratic Spline Surface Patch Helpers
# **************************************

def compute_normalized_surface_mapping(surface_mapping_coeffs: Matrix6x3r, domain: ConvexPolygon) -> Matrix6x3r:
    todo()
    return normalized_surface_mapping_coeffs


def compute_bezier_points(normalized_surface_mapping_coeffs: Matrix6x3r) -> Matrix6x3r:
    todo()
    return bezier_points


class QuadraticSplineSurfacePatch:
    """A quadratic surface patch with convex polygonal domain.

    Supports:
    - evaluation and sampling of points and normals on the surface
    - triangulation
    - conversion to Bezier form
    - bounding box computation
    - boundary curve parameterization
    - cone point annotation
    - (de)serialization
    """

    # **************
    # Public Methods
    # **************
    def __init__(self, surface_mapping_coeffs: Matrix6x3r | None = None,
                 domain: ConvexPolygon | None = None) -> None:

        # NOTE: domain is not set by the default constructor in ASOC code and can be None type
        if (surface_mapping_coeffs is None) and (domain is None):
            self.m_surface_mapping_coeffs: Matrix6x3r = np.zeros(shape=(6, 3))
            self.m_normal_mapping_coeffs: Matrix6x3r = np.zeros(shape=(6, 3))
            self.m_normalized_surface_mapping_coeffs: Matrix6x3r = np.zeros(
                shape=(6, 3))
            self.m_bezier_points: Matrix6x3r = np.zeros(shape=(6, 3))
            self.m_min_point: SpatialVector = np.zeros(shape=(1, 3))
            self.m_max_point: SpatialVector = np.zeros(shape=(1, 3))
            self.m_cone_index: int = -1
        elif (surface_mapping_coeffs and domain):
            # -- Core independent data --
            self.m_surface_mapping_coeffs: Matrix6x3r = surface_mapping_coeffs
            self.m_domain: ConvexPolygon = domain

            # -- Inferred dependent data --
            # Compute derived mapping information from the surface mapping and domain
            self.m_normal_mapping_coeffs: Matrix6x3r = generate_quadratic_surface_normal_coeffs(
                surface_mapping_coeffs)
            self.m_normalized_surface_mapping_coeffs: Matrix6x3r = compute_normalized_surface_mapping(
                surface_mapping_coeffs, domain)
            self.m_bezier_points: Matrix6x3r = compute_bezier_points(
                surface_mapping_coeffs)
            self.m_min_point, self.m_max_point = compute_point_cloud_bounding_box(
                self.m_bezier_points)

            # -- Additional cone marker to handle degenerate configurations --
            # NOTE: Do not mark a cone by default
            self.m_cone_index: int = -1
        else:
            unreachable(
                "Supposed to have either both surface_mapping_coeffs and domain or none for constructor")

    @property
    def dimension(self) -> int:
        """
        Get the dimension of the surface patch ambient space

        :return: dimension of the ambient space
        """
        return self.m_surface_mapping_coeffs.shape[1]

    def mark_cone(self, cone_index: int) -> None:
        """Mark one of the vertices as a cone

        :param cone_index: index of the cone in the triangle
        """
        self.m_cone_index = cone_index

    def has_cone(self) -> bool:
        """ Determine if the patch has a cone

        :return: true iff the patch has a cone
        :rtype: bool
        """
        return ((self.m_cone_index >= 0) and (self.m_cone_index < 3))

    def get_cone(self) -> int:
        """
        Get the cone index, or -1 if none exists.

        :return: true iff the patch has a cone
        :rtype: int
        """
        return self.m_cone_index

    def get_surface_mapping(self) -> Matrix6x3r:
        """
        Get the surface mapping coefficients.

        :return: reference to the surface mapping. shape==(6,3)
        :rtype: np.ndarray
        """
        return self.m_surface_mapping_coeffs

    def get_normal_mapping(self) -> Matrix6x3r:
        # TODO: change the names of all these getters into Python attribute equivalents
        """
        Get the surface normal mapping coefficients.

        :return: reference to the surface normal mapping. shape==(6,3)
        :rtype: np.ndarray
        """
        return self.m_normal_mapping_coeffs

    def get_normalized_surface_mapping(self) -> Matrix6x3r:
        """
        Get the surface mapping coefficients with normalized domain.

        Compute them if they haven't been computed yet.

        :return: reference to the normalized surface mapping. shape==(6,3)
        :rtype: np.ndarray
        """
        return self.m_normalized_surface_mapping_coeffs

    def get_bezier_points(self) -> Matrix6x3r:
        """
        Get the surface mapping coefficients with normalized domain.

        Compute them if they haven't been computed yet.

        :return: reference to the bezier points. shape==(6,3)
        :rtype: np.ndarray
        """
        return self.m_bezier_points

    def get_bounding_box(self) -> tuple[SpatialVector, SpatialVector]:
        """
        Compute the bounding box for the surface patch.

        :return: (self.m_min_point, self.m_max_point) where min_point is the minimum coordinates bounding box point and max_point is the maximum coordinates bounding box point.
        :rtype: tuple[SpatialVector, SpatialVector]
        """
        return self.m_min_point, self.m_max_point

    def get_bounding_box_min_point(self) -> SpatialVector:
        """
        Compute the minimum point of the bounding box for the surface patch.

        :return: min_point: minimum coordinates bounding box point
        :rtype: SpatialVector
        """
        return self.m_min_point

    def get_bbox_x_min(self) -> float:
        """
        Get the minimum x-coordinate of the bounding box.
        Note that min_point is NumPy shape (1, 3)

        :return: self.m_min_point[0][0]: x-coordinate of the minimum point of the bounding box
        :rtype: float
        """
        return self.m_min_point[0][0]

    def get_bbox_y_min(self) -> float:
        """
        Get the minimum y-coordinate of the bounding box.
        Note that min_point is NumPy shape (1, 3)

        :return: self.m_min_point[0][1]: y-coordinate of the minimum point of the bounding box
        :rtype: float
        """
        return self.m_min_point[0][1]

    def get_bounding_box_max_point(self) -> SpatialVector:
        """
        Compute the maximum point of the bounding box for the surface patch.

        :return: max_point: maximum coordinates bounding box point
        :rtype: SpatialVector
        """
        return self.m_max_point

    def get_bbox_x_max(self) -> float:
        """
        Get the maximum x-coordinate of the bounding box.
        Note that max_point is NumPy shape (1, 3)

        :return: self.m_max_point[0][0]: x-coordinate of the maximum point of the bounding box
        :rtype: float
        """
        return self.m_max_point[0][0]

    def get_bbox_y_max(self) -> float:
        """
        Get the maximum y-coordinate of the bounding box.

        :return: self.m_max_point[0][1]: y-coordinate of the maximum point of the bounding box
        :rtype: float
        """
        return self.m_max_point[0][1]

    @property
    def get_domain(self) -> ConvexPolygon:
        """
        Get the convex domain of the patch.

        :return: reference to the convex domain
        :rtype: ConvexPolygon
        """
        return self.m_domain

    def get_patch_boundaries(self) -> list[RationalFunction]:
        """
        Get the patch boundaries as spatial curves.

        :return: patch_boundaries: patch boundary spatial curves. degree=4, dimension=3
        :rtype: list[RationalFunction]
        """
        # Get parametrized domain boundaries.
        domain_boundaries: list[LineSegment] = self.get_domain.parametrize_patch_boundaries(
        )
        # Checking len == 3 since ASOC code has domain_boundarys as array of 3 LineSegment elements
        assert len(domain_boundaries) == 3

        # Lift the domain boundaries to the surface
        __ref_surface_mapping_coeffs: Matrix6x3r = self.get_surface_mapping()

        patch_boundaries: list[RationalFunction] = []

        # FIXME: Something might go wrong with the things below, especially since I'm unsure about surface_mapping_coeffs
        for i, domain_boundary in enumerate(domain_boundaries):
            patch_boundaries.append(domain_boundary.pullback_quadratic_function(
                3, __ref_surface_mapping_coeffs))

        assert len(patch_boundaries) == 3
        assert patch_boundaries[0].get_degree == 4
        assert patch_boundaries[0].get_dimension == 3

        return patch_boundaries

    def normalize_patch_domain(self) -> "QuadraticSplineSurfacePatch":
        """
        Construct a spline surface patch with the same image but where the domain
        is normalized to the triangle u + v <= 1 in the positive quadrant.

        :return: normalized_spline_surface_patch: normalized patch
        :rtype: QuadraticSplineSurfacePatch
        """
        # Generate the standard u + v <= 1 triangle
        # TODO: double check numpy array with the way eigen makes its matrices with the << operator
        normalized_domain_vertices: Matrix3x2r = np.array(
            [[0, 0], [1, 0], [0, 1]])
        assert normalized_domain_vertices.shape == (3, 2)
        normalized_domain: ConvexPolygon = ConvexPolygon.init_from_vertices(
            normalized_domain_vertices)

        # Build the normalized surface patch
        normalized_surface_mapping_coeffs: Matrix6x3r = self.get_normalized_surface_mapping()
        normalized_spline_surface_patch = QuadraticSplineSurfacePatch(
            normalized_surface_mapping_coeffs, normalized_domain)

        return normalized_spline_surface_patch

    def denormalize_domain_point(self, normalized_domain_point: PlanarPoint) -> PlanarPoint:
        """
        Given a normalized domain point in the triangle u + v <= 1, map it to the
        corresponding point in the patch domain.

        :param normalized_domain_point: normalized (barycentric) domain point
        :type normalized_domain_point: PlanarPoint

        :return: corresponding point in the domain triangle
        :rtype: PlanarPoint
        """
        # Replace with actual logic
        todo()
        # Get domain triangle vertices
        __ref_domain: ConvexPolygon = self.get_domain
        domain_vertices = __ref_domain.get_vertices
        v0: PlanarPoint = domain_vertices[[0], :]
        v1: PlanarPoint = domain_vertices[[1], :]
        v2: PlanarPoint = domain_vertices[[2], :]

        # Generate affine transformation mapping the standard triangle to the domain triangle
        linear_transformation = np.array([[v1 - v0], [v2 - v0]])
        assert linear_transformation.shape == (2, 2)
        translation: PlanarPoint = v0

        # Denormalize the domain point
        # shapes: (1, 2) @ (2, 2) + (1, 2)
        return normalized_domain_point @ linear_transformation + translation

    def evaluate(self, domain_point: PlanarPoint) -> SpatialVector:
        """
        Evaluate the surface at a given domain point.

        :param domain_point: domain evaluation point of shape (1, 2)
        :type domain_point: PlanarPoint

        :return: surface_point: image of the domain point on the surface of shape (1, 3)
        :rtype: SpatialVector
        """
        # TODO: double check that evaluate_quadratic_mapping returns something
        surface_point: SpatialVector = evaluate_quadratic_mapping(
            3, self.m_surface_mapping_coeffs, domain_point)
        assert surface_point.shape == (1, 3)
        return surface_point

    def evaluate_normal(self, domain_point: PlanarPoint) -> SpatialVector:
        """
        Evaluate the surface normal at a given domain point.

        :param domain_point: domain evaluation point of shape (1, 2)
        :type domain_point: PlanarPoint

        :return: surface_point: surface normal (of shape (1, 3)) at the image of the domain point
        :rtype: SpatialVector
        """
        surface_normal: SpatialVector = evaluate_quadratic_mapping(
            3, self.m_normal_mapping_coeffs, domain_point)

        assert surface_normal.shape == (1, 3)
        return surface_normal

    def sample(self, sampling_density: int) -> list[SpatialVector]:
        """
        Sample points on the surface.

        :param sampling_density: sampling density parameter
        :type sampling_density: int

        :return: spline_surface_patch_points: sampled points on the surface
        :rtype: list[SpatialVector]
        """
        # Sample the convex domain
        domain_points: list[PlanarPoint] = self.m_domain.sample(
            sampling_density)

        # Lift the domain points to the surface
        num_points: int = len(domain_points)

        spline_surface_patch_points: list[SpatialVector] = []

        # TODO: change for loop to utilize enumerate
        for i in range(num_points):
            spline_surface_patch_points.append(self.evaluate(domain_points[i]))

        return spline_surface_patch_points

    def triangulate(self,
                    num_refinements: int,
                    __ref_V: np.ndarray,
                    __ref_F: np.ndarray,
                    __ref_N: np.ndarray):
        # -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Triangulate the surface patch.

        :param num_refinements: number of refinements of the domain to perform.
        :type num_refinements: int

        # TODO: format params below to sphinx format
        @param[out] V: triangulated patch vertex positions (shape (n, 3))
        @param[out] F: triangulated patch faces 
        @param[out] N: triangulated patch vertex normals (shape (n, 3))

        # TODO: remove the return note because it is no longer true.
        :return: triangulated patch vertex positions (V), faces (F), and vertex normals (N)
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """

        # Triangulate the domain
        V_domain: np.ndarray
        F: np.ndarray
        # __ref_F: np.ndarray

        # TODO: will F be changed by reference or what? What will happen to F as it gets reassigned here?
        V_domain, F = self.m_domain.triangulate(num_refinements)

        # Lift the domain vertices to the surface and also compute the normals
        # reshape to (V_domain.rows(), self.dimension)
        __ref_V.reshape((V_domain.shape[0], self.dimension))
        __ref_N.reshape((V_domain.shape[0], self.dimension))

        for i in range(V_domain.shape[0]):  # V_domain.rows()
            # V_domain of shape
            surface_point: SpatialVector = self.evaluate(V_domain[[i], :])
            surface_normal: SpatialVector = self.evaluate_normal(V_domain[[i], :])

            # TODO: something might go wrong with the broadcasting shapes
            __ref_V[i, :] = surface_point.flatten()
            __ref_N[i, :] = surface_normal.flatten()

        # TODO: have the change in the method be reflected back into the parameter!
        todo("Have the changes to __ref_F be reflected OUTSIDE of the method since it's being  binded to the local np.ndarray and not modified by reference.")
        todo("So, have this return the triangulated V, F, and N arrays?")
        np.copyto(__ref_F, F)
        __ref_F.copy(F)

    def add_patch_to_viewer(self, patch_name: str = "surface_patch") -> None:
        """
        Add triangulated patch to the polyscope viewer.

        :param patch_name: name to assign the patch in the viewer.
        :type patch_name: str
        """
        # Generate mesh discretization
        num_refinements: int = 2

        # TODO: does the logic below work? Triangulate modifies by reference and that may bring up some issues with this Python code.
        V: np.ndarray = np.zeros(shape=(0, 0))
        F: np.ndarray = np.zeros(shape=(0, 0))
        N: np.ndarray = np.zeros(shape=(0, 0))

        self.triangulate(num_refinements, V, F, N)

        # Add patch mesh
        # TODO: does the below already add the patch to the viewer as it should or not?
        ps.init()
        ps_mesh = ps.register_surface_mesh(patch_name, V, F)

        # ps.show()

    def serialize(self) -> str:
        """
        Write the patch information to the output stream in the format
            c a_0 a_u a_v a_uv a_uu a_vv
            p1 p1_u p1_v
            p2 p2_u p2_v
            p3 p3_u p3_v

        :return: stream to write serialization to
        :rtype: str
        """
        todo()

    def write_patch(self, filename: str):
        """
        Write patch to file.

        :param filename: file to write serialized patch to.
        :type filename: str
        """
        todo()

    # ***************
    # Private methods
    # ***************
    def __formatted_patch(self):
        todo()
