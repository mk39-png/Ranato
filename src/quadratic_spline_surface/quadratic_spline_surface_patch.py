"""
Representation for quadratic surface patches with convex domains.
"""
from src.core.common import *
from src.core.convex_polygon import *
from src.core.evaluate_surface_normal import *
from src.core.polynomial_function import *
from src.core.rational_function import *

# TODO: have polyscope
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
    def __init__(self, surface_mapping_coeffs: Matrix6x3r | None = np.zeros(shape=(6, 3)),
                 normal_mapping_coeffs: Matrix6x3r = np.zeros(shape=(6, 3)),
                 normalized_surface_mapping_coeffs: Matrix6x3r = np.zeros(
                     shape=(6, 3)),
                 bezier_points: Matrix6x3r = np.zeros(shape=(6, 3)),
                 min_point: SpatialVector = np.zeros(shape=(1, 3)),
                 max_point: SpatialVector = np.zeros(shape=(1, 3)),
                 domain: ConvexPolygon | None = None) -> None:

        # Core independent data
        self.m_surface_mapping_coeffs: Matrix6x3r = surface_mapping_coeffs
        # NOTE: domain is not set by the default constructor in ASOC code and can be None type
        self.m_domain: ConvexPolygon = domain

        # Inferred dependent data
        self.m_normal_mapping_coeffs: Matrix6x3r
        self.m_normalized_surface_mapping_coeffs: Matrix6x3r
        self.m_bezier_points: Matrix6x3r
        self.m_min_point: SpatialVector
        self.m_max_point: SpatialVector

        # Additional cone marker to handle degenerate configurations
        # NOTE: Do not mark a cone by default
        self.m_cone_index: int = -1

    @property
    def dimension(self) -> int:
        """
        Get the dimension of the surface patch ambient space

        :return: dimension of the ambient space
        """
        todo()

    def mark_cone(self, cone_index: int) -> None:
        """Mark one of the vertices as a cone

        :param cone_index: index of the cone in the triangle
        """
        todo()

    def has_cone(self) -> bool:
        """ Determine if the patch has a cone

        :return: true iff the patch has a cone
        :rtype: bool
        """

    def get_cone(self) -> int:
        """
        Get the cone index, or -1 if none exists.

        :return: true iff the patch has a cone
        :rtype: int
        """
        pass

    def get_surface_mapping(self) -> "Matrix6x3r":
        """
        Get the surface mapping coefficients.

        :return: reference to the surface mapping. shape==(6,3)
        :rtype: np.ndarray
        """
        pass

    def get_normal_mapping(self) -> "Matrix6x3r":
        """
        Get the surface normal mapping coefficients.

        :return: reference to the surface normal mapping. shape==(6,3)
        :rtype: np.ndarray
        """
        pass

    def get_normalized_surface_mapping(self) -> "Matrix6x3r":
        """
        Get the surface mapping coefficients with normalized domain.

        Compute them if they haven't been computed yet.

        :return: reference to the normalized surface mapping. shape==(6,3)
        :rtype: np.ndarray
        """
        pass

    def get_bezier_points(self) -> "Matrix6x3r":
        """
        Get the surface mapping coefficients with normalized domain.

        Compute them if they haven't been computed yet.

        :return: reference to the bezier points. shape==(6,3)
        :rtype: np.ndarray
        """
        pass

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
        todo()

    def get_patch_boundaries(self) -> list[RationalFunction]:
        """
        Get the patch boundaries as spatial curves.

        :return: patch_boundaries: patch boundary spatial curves. degree=4, dimension=3
        :rtype: list[RationalFunction]
        """
        todo()

    def normalize_patch_domain(self) -> QuadraticSplineSurfacePatch:
        """
        Construct a spline surface patch with the same image but where the domain
        is normalized to the triangle u + v <= 1 in the positive quadrant.

        :return: normalized_spline_surface_patch: normalized patch
        :rtype: QuadraticSplineSurfacePatch
        """
        todo()

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

    def evaluate(self, domain_point: PlanarPoint) -> SpatialVector:
        """
        Evaluate the surface at a given domain point.

        :param domain_point: domain evaluation point
        :type domain_point: PlanarPoint

        :return: surface_point: image of the domain point on the surface
        :rtype: SpatialVector
        """
        todo("Maybe returning surface_point? Will have to see")

    def evaluate_normal(self, domain_point: PlanarPoint) -> SpatialVector:
        """
        Evaluate the surface normal at a given domain point.

        :param domain_point: domain evaluation point
        :type domain_point: PlanarPoint

        :return: surface_point: surface normal at the image of the domain point
        :rtype: SpatialVector
        """
        todo()

    def sample(self, sampling_density: int) -> list[SpatialVector]:
        """
        Sample points on the surface.

        :param sampling_density: sampling density parameter
        :type sampling_density: int

        :return: spline_surface_patch_points: sampled points on the surface
        :rtype: list[SpatialVector]
        """
        todo()

    def triangulate(self, num_refinements: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Triangulate the surface patch.

        :param num_refinements: number of refinements of the domain to perform.
        :type num_refinements: int

        :return: triangulated patch vertex positions (V), faces (F), and vertex normals (N)
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        todo()

    def add_patch_to_viewer(self, patch_name: str = "surface_patch") -> None:
        """
        Add triangulated patch to the polyscope viewer.

        :param patch_name: name to assign the patch in the viewer.
        :type patch_name: str
        """
        todo()

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
