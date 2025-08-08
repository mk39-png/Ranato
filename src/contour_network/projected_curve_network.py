"""
Methods to compute a simple planar curve network from annotated plane curve
soup.
"""


class ProjectedCurveNetwork(AbstractCurveNetwork):
    """
    A projected curve network is a curve network of intersecting planar curves
    arising from the projection of spatial curves to the xy plane.
    """

    def __init__(self) -> None:
        """
        Construct the curve network from the relevant annotated geometric
        information.
        @param[in] parameter_segments: uv domain quadratic curves parametrizing
            the other curves
        @param[in] spatial_segments: spatial rational curves before projection
        @param[in] planar_segments: planar rational curves after projection
        @param[in] chain_labels: list of maps of labels for each segment (e.g.,
        patch label)
        @param[in] chains: list of lists of chained curve indices
        @param[in] chain_labels: list of chain labels for each segment
        @param[in] interior_cusps: list of lists of cusp points per segment
        @param[in] has_cusp_at_base: list of bools per segment indicating if the
            segment base node is a cusp
        @param[in] intersections: list of lists of intersection points per segment
        @param[in] intersection_indices: list of lists of indices of curves
        corresponding
            to intersection points per segment
        """
