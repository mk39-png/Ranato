"""
Methods to compute a simple planar curve network from annotated plane curve
soup.
"""
import logging

from src.contour_network.intersection_data import IntersectionData
from src.core.abstract_curve_network import AbstractCurveNetwork
from src.core.common import (NodeIndex, SegmentIndex, logger, todo,
                             unimplemented)
from src.core.conic import Conic
from src.core.rational_function import RationalFunction
from src.utils.project_curve_networks_utils import (
    NodeGeometry, SegmentGeometry,
    build_projected_curve_network_without_intersections,
    connect_segment_intersections, remove_redundant_intersections,
    split_segments_at_cusps, split_segments_at_intersections)

# ***********************
# Projected Curve Network
# ***********************


class ProjectedCurveNetwork(AbstractCurveNetwork):
    """
    A projected curve network is a curve network of intersecting planar curves
    arising from the projection of spatial curves to the xy plane.
    :ivar segments: list[SegmentGeometry]
    :ivar nodes: list[NodeGeometry]
    :ivar chain_start_nodes: list[NodeIndex]
    """

    def __init__(self,
                 parameter_segments: list[Conic],
                 spatial_segments: list[RationalFunction],  # RationalFunction<4, 3>
                 planar_segments: list[RationalFunction],  # RationalFunction<4, 2>
                 segment_labels: list[dict[str, int]],
                 chains: list[list[int]],
                 chain_labels: list[int],
                 interior_cusps: list[list[float]],
                 has_cusp_at_base: list[bool],
                 intersections: list[list[float]],
                 intersection_indices: list[list[int]],
                 intersection_data: list[list[IntersectionData]],
                 num_intersections: int
                 ) -> None:
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
        self.__init_projected_curve_network(parameter_segments,
                                            spatial_segments,
                                            planar_segments,
                                            segment_labels,
                                            chains,
                                            chain_labels,
                                            interior_cusps,
                                            has_cusp_at_base,
                                            intersections,
                                            intersection_indices,
                                            intersection_data,
                                            num_intersections)

        if logger.getEffectiveLevel() == logging.DEBUG:
            # TODO: is the below protected or private?
            # NOTE: do not need to call the below because it's called upon super().__init__(...)
            # if not self._is_valid_abstract_curve_network():
            #     logger.error("Invalid abstract curve network made")
            #     raise RuntimeError("Invalid abstract curve network made")

            if not self._is_valid_projected_curve_network():
                logger.error("Invalid projected curve network made")
                raise RuntimeError("Invalid projected curve network made")

    # ******
    # Counts (inherited from parent class AbstractCurveNetworks)
    # ******
    # num_segments
    # num_nodes

    # ******
    # Geometry (setters and getters)
    # ******
    @property
    def nodes(self) -> list[NodeGeometry]:
        """Retrieves nodes of projected curve network"""
        return self.__nodes

    # @nodes.setter
    # def nodes(self, nodes: list[NodeGeometry]) -> None:
    #     self.__nodes = nodes

    @property
    def segments(self) -> list[SegmentGeometry]:
        """Retrieves segments of projected curve network"""
        return self.__segments

    # @segments.setter
    # def segments(self, segments: list[SegmentGeometry]) -> None:
    #     self.__segments = segments

    @property
    def chain_start_nodes(self) -> list[NodeIndex]:
        """Retrieves chain start nodes of projected curve network"""
        return self.__chain_start_nodes

    # ****************
    # Segment geometry
    # ****************
    # def segment_quantitative_invisibility(self, segment_index: SegmentIndex):

    # *************
    # Node geometry
    # *************

    def is_knot_node(self, node_index: NodeIndex) -> bool:
        """
        Used in SegmentChainIterator
        """
        unimplemented()

    def is_intersection_node(self, node_index) -> bool:
        """
        Checks self.__nodes at node_index if .is_intersection()
        """
        assert self._is_valid_node_index(node_index)
        if not self._is_valid_node_index(node_index):
            logger.error("Invalid node query")
            return False
        return self.__nodes[node_index].is_intersection()

    # ***************
    # Private Helpers
    # ***************

    def __init_projected_curve_network(self,
                                       parameter_segments: list[Conic],
                                       spatial_segments: list[RationalFunction],  # RationalFunction<4, 3>
                                       planar_segments: list[RationalFunction],  # RationalFunction<4, 2>
                                       segment_labels: list[dict[str, int]],
                                       chains: list[list[int]],
                                       chain_labels: list[int],
                                       interior_cusps: list[list[float]],
                                       has_cusp_at_base: list[bool],
                                       intersections: list[list[float]],
                                       intersection_indices: list[list[int]],
                                       intersection_data_ref: list[list[IntersectionData]],
                                       num_intersections: int) -> None:
        """
        Main constructor implementation
        TODO: inherit docstring from constructor
        """
        num_segments: int = len(planar_segments)

        if len(parameter_segments) != num_segments:
            logger.error("Inconsistent number of segments")
        if len(spatial_segments) != num_segments:
            logger.error("Inconsistent number of segments")
        if len(planar_segments) != num_segments:
            logger.error("Inconsistent number of segments")
        if len(segment_labels) != num_segments:
            logger.error("Inconsistent number of segments")
        if len(chain_labels) != num_segments:
            logger.error("Inconsistent number of segments")
        if len(interior_cusps) != num_segments:
            logger.error("Inconsistent number of segments")
        if len(has_cusp_at_base) != num_segments:
            logger.error("Inconsistent number of segments")
        if len(intersections) != num_segments:
            logger.error("Inconsistent number of segments")
        if len(intersection_indices) != num_segments:
            logger.error("Inconsistent number of segments")
        logger.info("Building projected curve network for %s segments",
                    num_segments)

        # Connect segments into chains before splitting at intersections
        to_array: list[NodeIndex]
        out_array: list[SegmentIndex]
        self.__segments: list[SegmentGeometry]
        self.__nodes: list[NodeGeometry]
        (to_array,
         out_array,
         self.__segments,
         self.__nodes) = build_projected_curve_network_without_intersections(
            parameter_segments,
            spatial_segments,
            planar_segments,
            segment_labels,
            chains,
            has_cusp_at_base)
        assert self._is_valid_curve_data(to_array, out_array)
        self.__mark_open_chain_endpoints(to_array, out_array, chains, self.nodes)

        # Remove intersections that are redundant
        remove_redundant_intersections(to_array,
                                       out_array,
                                       num_intersections,
                                       intersection_data_ref)
        assert self._is_valid_curve_data(to_array, out_array)

        # Split segments at intersections while maintaining a record of the original
        # segments
        original_segment_indices: list[SegmentIndex]
        split_segment_indices: list[list[SegmentIndex]]
        intersection_nodes: list[list[NodeIndex]]

        # TODO: make below more Pythonic. to_array, out_array, __segments, and __nodes
        # are modified by reference
        (original_segment_indices,
         split_segment_indices,
         intersection_nodes) = split_segments_at_intersections(
            intersection_data_ref,
            num_intersections,
            to_array,
            out_array,
            self.__segments,
            self.__nodes,
        )

        assert self._is_valid_curve_data(to_array, out_array)

        for i, _ in enumerate(intersection_nodes):
            logger.info("Intersection %s: %s", i, intersection_nodes[i])
            if (len(intersection_nodes[i]) != 2) and (len(intersection_nodes[i]) != 0):
                logger.warning("Intersection %s does not have two nodes: %s",
                               i,
                               intersection_nodes[i])

        # Link intersection nodes

        # Initialize all intersection indices to -1
        num_nodes: int = len(out_array)
        intersection_array: list[NodeIndex] = [-1] * num_nodes

        # NOTE: below modifies intersection_array and self.__nodes by reference.
        connect_segment_intersections(self.segments,
                                      intersection_data_ref,
                                      intersection_nodes,
                                      to_array,
                                      out_array,
                                      split_segment_indices,
                                      intersection_array,
                                      self.__nodes)

        assert self._is_valid_minimal_curve_network_data(to_array, out_array, intersection_array)

        # Further split segments at cusps
        split_segments_at_cusps(interior_cusps,
                                original_segment_indices,
                                split_segment_indices,
                                to_array,
                                out_array,
                                intersection_array,
                                self.__segments,
                                self.__nodes)

        assert self._is_valid_minimal_curve_network_data(to_array, out_array, intersection_array)

        # Rebuild topology with intersection and cusp splits
        self.update_topology(to_array, out_array, intersection_array)

        # Record chain start points
        self.__init_chain_start_nodes()

        # Check validity
        for node_index in range(self.num_nodes):
            if (self.is_intersection_node(node_index) and
                    not self._is_valid_node_index(self.intersection(node_index))):
                raise ValueError(f"Intersection node {node_index} does not \
                                  have a valid intersection")

    def __init_chain_start_nodes(self) -> None:
        """
        Add all special nodes except the path end nodes to the list of chain start
        nodes
        WARNING: This method is a little dangerous; it modifies the segments as it
        iterates over them
        """
        num_nodes: NodeIndex = len(self.__nodes)
        self.__chain_start_nodes: list[NodeIndex] = []
        # Get all nodes that are special (and not path end nodes)
        # self.__chain_start_nodes
        for ni in range(num_nodes):
            if not self.__nodes[ni].is_knot() and not self.__nodes[ni].is_path_end_node():
                if self.out(ni) < 0:
                    continue  # Hack to skip intersection path end nodes
                self.__chain_start_nodes.append(ni)

        all_nodes_covered = False
        while not all_nodes_covered:
            # Get list of all covered nodes
            is_covered_node: list[bool] = [False] * num_nodes
            for i, ni in enumerate(self.__chain_start_nodes):
                is_covered_node[ni] = True
                start_si: SegmentIndex = self.out(ni)
                if not self._is_valid_segment_index(start_si):
                    raise ValueError("Start node is an end point")

                # Check chain from start node
                # FIXME: potential mistranslation from C++ to Python (esp w/ pointers)
                # iter: SegmentChainIterator = self._get_segment_chain_iterator(start_si)
                for si in range(start_si,):
                    is_covered_node[self.to(si)] = True

        todo("Fix the implementation of this method properly...")
        # return chain_start_nodes

    def __mark_open_chain_endpoints(self,
                                    to_array: list[NodeIndex],
                                    out_array: list[SegmentIndex],
                                    chains: list[list[SegmentIndex]],
                                    nodes_ref: list[NodeGeometry]) -> None:
        """
        Record the start of open chains and also mark an arbitrary node on each
        closed contour
        """
        # Build from array from the network topology
        from_array: list[NodeIndex] = self.build_from_array(to_array, out_array)

        for i, _ in enumerate(chains):
            # Get the first and last segments in the chain
            first_segment: SegmentIndex = chains[i][0]
            last_segment: SegmentIndex = chains[i][-1]

            if to_array[last_segment] != from_array[first_segment]:
                start_node: NodeIndex = from_array[first_segment]
                end_node: NodeIndex = to_array[last_segment]
                nodes_ref[start_node].mark_as_path_start_node()
                nodes_ref[end_node].mark_as_path_end_node()
