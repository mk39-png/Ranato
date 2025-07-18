"""
Convex polygons. Used in Quadratic Surface Patch files.
Convex polygon formed by intersecting half planes.
"""

from src.core.line_segment import LineSegment
from src.core.common import *  # import float_equal, generate_linspace, PlanarPoint, Index
from src.core.interval import *
import numpy as np


# *******
# Helpers
# *******

def compute_line_between_points(point_0: PlanarPoint, point_1: PlanarPoint) -> np.ndarray:
    """
    Compute the implicit form of a line between two points

    :param point_0: first point of shape (1, 2)
    :type point_0: PlanarPoint
    :param point_1: second point of shape (1, 2)
    :type point_1: PlanarPoint

    :return: line_coeff of shape (3, 1) made from the points
    :rtype: np.ndarray 
    """
    x0 = point_0[0, 0]
    y0 = point_0[0, 1]
    x1 = point_1[0, 0]
    y1 = point_1[0, 1]
    line_coeffs: np.ndarray = np.array([[x0 * y1 - x1 * y0],
                                        [y0 - y1],
                                        [x1 - x0]])

    assert line_coeffs.shape == (3, 1)
    return line_coeffs


def compute_parametric_line_between_points(point_0: PlanarPoint,
                                           point_1: PlanarPoint) -> LineSegment:
    """
    Compute the parametric form of a line between two points

    :param point_0: first point of shape (1, 2)
    :type point_0: PlanarPoint
    :param point_1: second point of shape (1, 2)
    :type point_1: PlanarPoint

    :return: line_segment
    :rtype: LineSegment
    """
    # Set numerator
    numerators: np.ndarray = np.array([
        [point_0[0, 0], point_0[0, 1]],
        [point_1[0, 0] - point_0[0, 0], point_1[0, 1] - point_0[0, 1]]])
    # TODO: double check that the elements in numerators are indeed as per ASOC code
    # numerators(0, 0) = point_0(0)
    # numerators(0, 1) = point_0(1)
    # numerators(1, 0) = point_1(0) - point_0(0)
    # numerators(1, 1) = point_1(1) - point_0(1)

    assert numerators.shape == (2, 2)

    # Set domain interval [0, 1]
    # TODO: below may not be the most Python way of making the Interval object domain
    domain: Interval = Interval()
    domain.set_lower_bound(0, False)
    domain.set_upper_bound(1, False)

    line_segment = LineSegment(numerators, domain)
    return line_segment


def refine_triangles(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Refine a mesh with midpoint subdivision.
    Logic of this method does not modify V and F by reference and instead creates new np.ndarray V_refined and F_refined.

    :param V: vertices
    :type V: np.ndarray

    :param F: faces
    :type F: np.ndarray

    :return: Vertices and Faces refined
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    assert V.dtype == np.float64
    assert F.dtype == np.int64

    num_faces: Index = F.shape[0]  # rows

    V_refined: np.ndarray = np.ndarray(shape=(num_faces * 6, 2))
    F_refined: np.ndarray = np.ndarray(shape=(num_faces * 4, 3))

    # TODO: could probably use NumPy indexing for the things below, right?
    for i in range(num_faces):
        # We have vectors below, this time of shape (n, ) because we're using NumPy broadcasting
        v0: np.ndarray = V[F[i, 0], :]
        v1: np.ndarray = V[F[i, 1], :]
        v2: np.ndarray = V[F[i, 2], :]
        assert v0.ndim == 1
        assert v1.ndim == 1
        assert v2.ndim == 1

        # Add vertices for refined face
        # TODO: do I need ", :" to access the row?
        V_refined[6 * i + 0, :] = v0
        V_refined[6 * i + 1, :] = v1
        V_refined[6 * i + 2, :] = v2
        V_refined[6 * i + 3, :] = (v0 + v1) / 2.0
        V_refined[6 * i + 4, :] = (v1 + v2) / 2.0
        V_refined[6 * i + 5, :] = (v2 + v0) / 2.0

        # Add refined faces
        F_refined[4 * i + 0, :] = np.array([6 * i + 0, 6 * i + 3, 6 * i + 5])
        F_refined[4 * i + 1, :] = np.array([6 * i + 1, 6 * i + 4, 6 * i + 3])
        F_refined[4 * i + 2, :] = np.array([6 * i + 2, 6 * i + 5, 6 * i + 4])
        F_refined[4 * i + 3, :] = np.array([6 * i + 3, 6 * i + 4, 6 * i + 5])

    return V_refined, F_refined


class ConvexPolygon:
    """
    Representation of a convex polygon in R^2 that supports containment queries,
    sampling, boundary segments and vertices computation, triangulation, and
    boundary parametrization.
    """
    # TODO (from ASOC code): Implement constructor from collection of points

    def __init__(self, boundary_segments_coeffs: list[np.ndarray],
                 vertices: np.ndarray) -> None:
        """
        Constructor that is called by classmethod init_from_boundary_segments_coeffs or init_from_vertices.
        NOTE: Do not call this constructor directly. As in, do not call ConvexPolygon(boundary_segments_coeffs, vertices). Instead, use ConvexPolygon.init_from_boundary_segments_coeffs or ConvexPolygon.init_from_vertices().

        :param boundary_segments_coeffs: boundary segment coefficients list of size 3 with element np.ndarray of shape (3, 1) and type np.float64
        :type boundary_segments_coeffs: list[np.ndarray]
        :param vertices: vertices of type np.ndarray of shape (3, 2)
        :type vertices: np.ndarray
        """
        # Assertions to match ASOC code C++ code
        assert len(boundary_segments_coeffs) == 3
        assert boundary_segments_coeffs[0].shape == (3, 1)
        assert vertices.shape == (3, 2)

        # *******
        # Private
        # *******
        self.m_boundary_segments_coeffs: list[np.ndarray] = boundary_segments_coeffs
        self.m_vertices: np.ndarray = vertices

    @classmethod
    def init_from_boundary_segments_coeffs(cls, boundary_segments_coeffs: list[np.ndarray]):
        """
        Only boundary_segments_coeffs passed in. Construct m_vertices.
        """
        v0: PlanarPoint = cls.intersect_patch_boundaries(boundary_segments_coeffs[1], boundary_segments_coeffs[2])
        v1: PlanarPoint = cls.intersect_patch_boundaries(boundary_segments_coeffs[2], boundary_segments_coeffs[0])
        v2: PlanarPoint = cls.intersect_patch_boundaries(boundary_segments_coeffs[0], boundary_segments_coeffs[1])

        # TODO: may be problem with vertices shape
        # v0 shape == (1, 2)
        # v1 shape == (1, 2)
        # v2 shape == (1, 2)
        # So, we want vertices shape == (3, 2)
        vertices: np.ndarray = np.array([v0, v1, v2])
        assert vertices.shape == (3, 2)

        # return vertices
        return cls(boundary_segments_coeffs, vertices)

    @classmethod
    def init_from_vertices(cls, vertices: np.ndarray):
        """
        Only vertices passed in. Construct m_boundary_segments_coeffs.
        """
        assert vertices.shape == (3, 2)
        num_vertices: int = vertices.shape[0]
        boundary_segments_coeffs: list[np.ndarray] = []

        # TODO: is the below the dynamic sizing of arrays that I needed to avoid?
        for i in range(num_vertices):
            line_coeffs: np.ndarray = compute_line_between_points(vertices[[i], :],
                                                                  vertices[[(i + 1) % num_vertices], :])
            boundary_segments_coeffs.append(line_coeffs)

        assert len(boundary_segments_coeffs) == 3
        assert boundary_segments_coeffs[0].shape == (3, 1)

        # return boundary_segments_coeffs
        return cls(boundary_segments_coeffs, vertices)

    def contains(self, point: PlanarPoint) -> bool:
        """
        Return true iff point is in the convex polygon
        """
        for i, L_coeffs in enumerate(self.m_boundary_segments_coeffs):
            # NOTE: redundant check
            assert L_coeffs.shape == (3, 1)

            if (L_coeffs[0, 0] + L_coeffs[1, 0] * point[0, 0] + L_coeffs[2, 0] * point[1, 0]) < 0.0:
                return False

        return True

    @staticmethod
    def intersect_patch_boundaries(first_boundary_segment_coeffs: np.ndarray,
                                   second_boundary_segment_coeffs: np.ndarray) -> PlanarPoint:
        """
        NOTE: method has decorator @staticmethod to work with @classmethod init_from_boundary_segments_coeffs.
        """
        assert first_boundary_segment_coeffs.shape == (3, 1)
        assert second_boundary_segment_coeffs.shape == (3, 1)

        a00: float = first_boundary_segment_coeffs[0, 0]
        a10: float = first_boundary_segment_coeffs[1, 0]
        a01: float = first_boundary_segment_coeffs[2, 0]
        b00: float = second_boundary_segment_coeffs[0, 0]
        b10: float = second_boundary_segment_coeffs[1, 0]
        b01: float = second_boundary_segment_coeffs[2, 0]

        x: float
        y: float

        # Solve for y in terms of x first
        if not float_equal(a01, 0.0):
            my: float = -a10 / a01
            by: float = -a00 / a01
            assert not float_equal(b10 + b01 * my, 0.0)
            x = -(b00 + b01 * by) / (b10 + b01 * my)
            y = my * x + by
        # Solve for x in terms of y first
        elif not float_equal(a10, 0.0):
            mx: float = -a01 / a10
            bx: float = -a00 / a10
            assert not float_equal(b01 + b10 * mx, 0.0)
            y = -(b00 + b10 * bx) / (b01 + b10 * mx)
            x = mx * y + bx
        else:
            logger.error("Degenerate line")
            # TODO: maybe exception is a bit too extreme... originally there was a "return"
            # here in the ASOC code
            unreachable("Error with intersect_patch_boundaries()")

        # Build intersection
        # TODO: I really need a subclass of np.ndarray called PlanarPoint that automatically checks
        # the sizing of the array for me... Because manually checking shape==(1,2) is cumbersome
        intersection = np.array([[x], [y]])
        assert intersection.shape == (1, 2)

        return intersection

    @property
    # TODO: change the name
    def get_boundary_segments(self) -> list[np.ndarray]:
        return self.m_boundary_segments_coeffs

    @property
    def get_vertices(self) -> np.ndarray:
        return self.m_vertices

    def parametrize_patch_boundaries(self) -> list[LineSegment]:
        patch_boundaries: list[LineSegment] = []

        # Get rows of m_vertices
        num_vertices = self.m_vertices.shape[0]

        for i in range(num_vertices):
            # TODO: maybe something wrong with the shape?
            # Because we need to grab shape (1, 2) from m_vertices
            # m_vertices is shape (3, 2)...
            line_segment: LineSegment = compute_parametric_line_between_points(
                self.m_vertices[[i], :],
                self.m_vertices[[(i + 1) % num_vertices], :])
            patch_boundaries.append(line_segment)

        # Double checking that we indeed only have 3 elements inside patch_boundaries as per
        # ASOC code
        assert len(patch_boundaries) == 3
        return patch_boundaries

    def triangulate(self, num_refinements: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Triangulate domain with.
        This takes in self.m_vertices and creates a new F matrix of shape (1, 3) to return.
        TODO Can generalize to arbitrary domain if needed
        """
        V: np.ndarray = self.m_vertices
        F: np.ndarray = np.array([[0, 1, 2]])
        assert F.shape == (1, 3)

        for i in range(num_refinements):
            V_refined: np.ndarray
            F_refined: np.ndarray
            V_refined, F_refined = refine_triangles(V, F)
            V = V_refined
            F = F_refined

        return V, F

    def sample(self, num_samples: int) -> list[PlanarPoint]:
        domain_points: list[PlanarPoint] = []

        # TODO (from ASOC code): Make actual bounding box
        lower_left_corner: PlanarPoint = np.array([[-1, -1]])
        upper_right_corner: PlanarPoint = np.array([[1, 1]])

        # Checking if shape (1, 2) since that is the shape of PlanarPoint type
        assert lower_left_corner.shape == (1, 2)
        assert upper_right_corner.shape == (1, 2)

        x0: float = lower_left_corner[0, 0]
        y0: float = lower_left_corner[0, 1]
        x1: float = upper_right_corner[0, 0]
        y1: float = upper_right_corner[0, 1]

        # Compute points
        x_axis: VectorX = generate_linspace(x0, x1, num_samples)
        y_axis: VectorX = generate_linspace(y0, y1, num_samples)
        # Asserting ndim == 1 because ASOC code has x_axis and y_axis as VectorXr
        assert x_axis.ndim == 1
        assert y_axis.ndim == 1

        for i in range(num_samples):
            for j in range(num_samples):
                point: PlanarPoint = np.array([[x_axis[i], y_axis[j]]])
                assert point.shape == (1, 2)
                if self.contains(point):
                    domain_points.append(point)

        return domain_points
