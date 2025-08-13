"""
File separate from project_curve_networks that holds helper classes and methods with the 
aim to shorten the length of the original file.
"""

from collections import defaultdict
from enum import Enum

from src.contour_network.intersection_data import IntersectionData
# from src.contour_network.projected_curve_network import ProjectedCurveNetwork
# from src.contour_network.projected_curve_network import ProjectedCurveNetwork
from src.core.abstract_curve_network import AbstractCurveNetwork
from src.core.common import (NodeIndex, SegmentIndex, arange, logger,
                             vector_equal)
from src.core.conic import Conic
from src.core.rational_function import RationalFunction

# **************
# Helper Classes
# **************


class SegmentGeometry():
    """Geometry at segment"""

    def __init__(self,
                 parameter_curve: Conic,
                 spatial_curve: RationalFunction,  # <4, 3>
                 planar_curve: RationalFunction,  # <4, 2>
                 segment_labels: dict[str, int]) -> None:
        """Constructor for SegmentGeometry"""
        assert spatial_curve.degree == 4
        assert spatial_curve.dimension == 3
        assert planar_curve.degree == 4
        assert planar_curve.dimension == 2

        self.__planar_curve: RationalFunction = planar_curve
        self.__spatial_curve: RationalFunction = spatial_curve
        self.__parameter_curve: Conic = parameter_curve
        self.__segment_labels: dict[str, int] = segment_labels
        self.__quantitative_invisibility = -1

        if not self.__is_valid_segment_geometry():
            raise ValueError("Invalid segment made")

    # *******************
    # Getters and setters
    # *******************

    @property
    def parameter_curve(self) -> Conic:
        """Retrieves parameter curve of type Conic"""
        return self.__parameter_curve

    @property
    def spatial_curve(self) -> RationalFunction:
        """Retrieves spatial curve of type RationalFunction. degree = 4, dimension = 3"""
        return self.__spatial_curve

    @property
    def planar_curve(self) -> RationalFunction:
        """Retrieves planar curve. degree = 4, dimension = 3"""
        return self.__planar_curve

    def set_segment_label(self, label_name: str, new_segment_label: int) -> None:
        """Sets segment_label at key label_name with value new_segment_label"""
        self.__segment_labels[label_name] = new_segment_label

    def get_segment_label(self, label_name: str) -> NodeIndex:
        """Retrives segment label at label_name key. Raises value error if key does not exist"""
        segment_label: NodeIndex | None = self.__segment_labels.get(label_name)
        if segment_label is None:
            raise ValueError(f"{label_name} does not exist in segment_labels")

        return segment_label

    @property
    def quantitative_invisibility(self) -> NodeIndex:
        """Retrieves quantitative invisibility"""
        return self.__quantitative_invisibility

    @quantitative_invisibility.setter
    def set_quantitative_invisibility(self, new_quantitative_invisibility: int) -> None:
        """Sets quantitative_invisibility"""
        if new_quantitative_invisibility < 0:
            raise ValueError("Cannot set negative segment Q1")
        self.__quantitative_invisibility: NodeIndex = new_quantitative_invisibility

    # ****************
    # Utility methods
    # ***************
    def split_at_knot(self,
                      knot: float) -> tuple["SegmentGeometry", "SegmentGeometry"]:
        """
        Split segment into two new segments at a given knot value

        :param knot: knot value to split at
        :type knot: float
        :return lower_segment: lower segment from split curve at knot
        :return upper_segment: upper segment from split curve at knot
        """
        # Split all three segment curves at the knot
        lower_parameter_curve: Conic
        lower_spatial_curve: RationalFunction  # <4, 3>
        lower_planar_curve: RationalFunction  # <4, 2>
        upper_parameter_curve: Conic
        upper_spatial_curve: RationalFunction  # <4, 3>
        upper_planar_curve: RationalFunction  # <4, 2>

        # TODO: fix typing issue below since Conic inherits split_at_knot() from RationalFunction
        lower_parameter_curve, upper_parameter_curve = self.parameter_curve.split_at_knot(knot)
        lower_planar_curve, upper_planar_curve = self.planar_curve.split_at_knot(knot)
        lower_spatial_curve, upper_spatial_curve = self.spatial_curve.split_at_knot(knot)

        # Build new segments from the split curves
        lower_segment = SegmentGeometry(lower_parameter_curve,
                                        lower_spatial_curve,
                                        lower_planar_curve,
                                        self.__segment_labels)
        upper_segment = SegmentGeometry(upper_parameter_curve,
                                        upper_spatial_curve,
                                        upper_planar_curve,
                                        self.__segment_labels)

        return lower_segment, upper_segment

    # ***************
    # Private Methods
    # ***************

    def __str__(self) -> str:
        string_print: str = f"""
        Parameter curve segment: {self.parameter_curve}\n
        Spatial curve segment: {self.spatial_curve}\n
        Planar curve segment: {self.planar_curve}\n
        """
        return string_print

    def __is_valid_segment_geometry(self) -> bool:
        """Verifies validity of segment geometry"""
        t0: float = self.planar_curve.domain.lower_bound
        t1: float = self.planar_curve.domain.upper_bound

        if self.spatial_curve.domain.lower_bound != t0:
            logger.error("Lower bound error")
            return False
        if self.parameter_curve.domain.lower_bound != t0:
            logger.error("Lower bound error")
            return False
        if self.spatial_curve.domain.upper_bound != t1:
            logger.error("Upper bound error")
            return False
        if self.parameter_curve.domain.upper_bound != t1:
            logger.error("Upper bound error")
            return False

        return True


class GeometricData(Enum):
    """Used by NodeGeometry class"""
    # TODO boundary and interior cusps are no longer clear in the general
    # context. Should rename to something more general like C1 cusp and C0
    # cusp, etc.
    KNOT = 0
    MARKED_KNOT = 1
    BOUNDARY_CUSP = 2
    INTERIOR_CUSP = 3
    INTERSECTION = 4
    PATH_END_NODE = 5
    PATH_START_NODE = 6


class NodeGeometry():
    """Geometry at node"""

    def __init__(self) -> None:
        """Constructor for NodeGeometry"""
        self.__node_type: GeometricData = GeometricData.KNOT
        self.__quantitative_invisibility: int = -1
        self.__qi_set = False

    def mark_as_knot(self) -> None:
        """Mark node type as KNOT enum"""
        self.__node_type = GeometricData.KNOT

    def mark_as_marked_knot(self) -> None:
        """Mark node type as MARKED_KNOT enum"""
        self.__node_type = GeometricData.MARKED_KNOT

    def mark_as_boundary_cusp(self) -> None:
        """Mark node type as BOUNDARY_CUSP enum"""
        self.__node_type = GeometricData.BOUNDARY_CUSP

    def mark_as_interior_cusp(self) -> None:
        """Mark node type as INTERIOR_CUSP enum"""
        self.__node_type = GeometricData.INTERIOR_CUSP

    def mark_as_intersection(self) -> None:
        """Mark node type as INTERSECTION enum"""
        self.__node_type = GeometricData.INTERSECTION

    def mark_as_path_start_node(self) -> None:
        """Mark node type as PATH_START_NODE enum"""
        self.__node_type = GeometricData.PATH_START_NODE

    def mark_as_path_end_node(self) -> None:
        """Mark node type as PATH_END_NODE enum"""
        self.__node_type = GeometricData.PATH_END_NODE

    def is_knot(self) -> bool:
        """Checks if node_type is KNOT enum"""
        return self.__node_type == GeometricData.KNOT

    def is_marked_knot(self) -> bool:
        """Checks if node_type is MARKED_KNOT enum"""
        return self.__node_type == GeometricData.MARKED_KNOT

    def is_boundary_cusp(self) -> bool:
        """Checks if node_type is BOUNDARY_CUSP enum"""
        return self.__node_type == GeometricData.BOUNDARY_CUSP

    def is_interior_cusp(self) -> bool:
        """Checks if node_type is INTERIOR_CUSP enum"""
        return self.__node_type == GeometricData.INTERIOR_CUSP

    def is_intersection(self) -> bool:
        """Checks if node_type is INTERSECTION enum"""
        return self.__node_type == GeometricData.INTERSECTION

    def is_path_start_node(self) -> bool:
        """Checks if node_type is PATH_START_NODE enum"""
        return self.__node_type == GeometricData.PATH_START_NODE

    def is_path_end_node(self) -> bool:
        """Checks if node_type is PATH_END_NODE enum"""
        return self.__node_type == GeometricData.PATH_END_NODE

    @property
    def quantitative_invisibility(self) -> NodeIndex:
        """Retrieves quantitative invisibility"""
        return self.__quantitative_invisibility

    @quantitative_invisibility.setter
    def set_quantitative_invisibility(self, new_quantitative_invisibility: int) -> None:
        """Sets quantitative_invisibility"""
        self.__qi_set = True
        if new_quantitative_invisibility < 0:
            raise ValueError("Cannot set negative segment Q1")
        self.__quantitative_invisibility: NodeIndex = new_quantitative_invisibility

    def quantitative_invisibility_is_set(self) -> bool:
        """Returns self.__qi_set"""
        return self.__qi_set

    def mark_quantitative_invisibility_as_set(self) -> None:
        """Marks quantitative invisibility as True"""
        self.__qi_set = True

    def formatted_node(self) -> str:
        """Readable representation of NodeGeometry"""
        if self.__node_type == GeometricData.KNOT:
            return "knot"
        elif self.__node_type == GeometricData.MARKED_KNOT:
            return "marked_knot"
        elif self.__node_type == GeometricData.BOUNDARY_CUSP:
            return "boundary_cusp"
        elif self.__node_type == GeometricData.INTERIOR_CUSP:
            return "interior_cusp"
        elif self.__node_type == GeometricData.INTERSECTION:
            return "intersection"
        elif self.__node_type == GeometricData.PATH_END_NODE:
            return "path_end_node"
        elif self.__node_type == GeometricData.PATH_START_NODE:
            return "path_start_node"
        else:
            return "unknown"


# class SegmentChainIterator():
#     """
#     Iterator over contour segment chains until FIXME is found.
#     """

#     def __init__(self, parent: ProjectedCurveNetwork, segment_index: SegmentIndex) -> None:
#         """
#         Constructor for SegmentChainIterator
#         """
#         self.__parent: ProjectedCurveNetwork = parent
#         self.__current_segment_index: SegmentIndex = segment_index
#         self.__is_end_of_chain = False
#         self.__is_reverse_end_of_chain = False
#         logger.info("Iterator initialized to segment %s", self.__current_segment_index)


#     def get_segment_chain_iterator(self, segment_index: SegmentIndex):
#         """
#         Build a segment chain iterator from some starting segment index.
#         @param segment_index: index to start the chain iterator at
#         @return chain iterator for the given starting segment
#         """
#         assert self.


# ***************
# Helper Function
# ***************


def build_projected_curve_network_without_intersections(parameter_segments: list[Conic],
                                                        # RationalFunction<4, 3>
                                                        spatial_segments: list[RationalFunction],
                                                        # RationalFunction<4, 2>
                                                        planar_segments: list[RationalFunction],
                                                        segment_labels: list[dict[str, int]],
                                                        chains: list[list[int]],
                                                        has_cusp_at_base: list[bool],
                                                        ) -> tuple[list[NodeIndex],
                                                                   list[SegmentIndex],
                                                                   list[SegmentGeometry],
                                                                   list[NodeGeometry]]:
    """
    Initialize the curve network without intersections directly from the input
    curve data, marking cusps between boundaries.
    FIXME Rename to indicate no cusps either

    :return: to_array
    :return: out_array
    :return: segments
    :return: nodes
    """
    num_segments: int = len(spatial_segments)
    to_array: list[NodeIndex] = []
    out_array: list[SegmentIndex] = []
    segments: list[SegmentGeometry] = []
    nodes: list[NodeGeometry] = []

    # Initialize segments without intersections
    logger.info("Initializing segment geometry without intersections")
    for i in range(num_segments):
        segments.append(SegmentGeometry(parameter_segments[i],
                                        spatial_segments[i],
                                        planar_segments[i],
                                        segment_labels[i]))

    # Initializing node bases without intersections
    logger.info("Initializing node bases without intersections")
    for i in range(num_segments):
        nodes.append(NodeGeometry())
        if has_cusp_at_base[i]:
            logger.info("Marking node %s as boundary cusp", i)
            nodes[i].mark_as_boundary_cusp()

    # Initialize one node for the base of each segment. Since we assume all
    # vertices are degree 2 or 1, there are always as many nodes as segments.
    out_array = arange(num_segments)

    # Connect the tips of segments to base nodes at the same location in space or
    # cap them with new nodes if no such base node exists. WARNING: Assumes the
    # base of the segment has the same index
    to_array = [-1] * num_segments
    for i, _ in enumerate(chains):
        for j, _ in enumerate(chains[i], 1):
            to_array[chains[i][j - 1]] = chains[i][j]

        # Close vector if the curve is closed or add a cap node otherwise
        first_segment: SegmentIndex = chains[i][0]
        last_segment: SegmentIndex = chains[i][-1]
        if vector_equal(spatial_segments[first_segment].start_point(),
                        spatial_segments[last_segment].end_point(),
                        1e-6):
            to_array[last_segment] = first_segment
            out_array[to_array[last_segment]] = first_segment
        else:
            # Add a node with no outgoing vector
            to_array[last_segment] = len(out_array)
            out_array.append(-1)
            nodes.append(NodeGeometry())
            nodes[-1].mark_as_path_end_node()

    return to_array, out_array, segments, nodes


def remove_redundant_intersections(to_array: list[NodeIndex],
                                   out_array: list[SegmentIndex],
                                   num_intersections: int,
                                   intersection_data_ref: list[list[IntersectionData]]) -> None:
    """
    Remove intersections between adjacent segments at the tip/base of the
    contours and one of two intersections if it is at the respective tip and base
    of two adjacent segments
    """
    # Build map from intersection ids to their index in intersection data
    intersection_segments: dict[int, dict[int, tuple[int, int]]] = defaultdict(dict)
    for i, _ in enumerate(intersection_data_ref):
        for j, _ in enumerate(intersection_data_ref[i]):
            intersection_id: int = intersection_data_ref[i][j].id
            intersection_segments[intersection_id][j] = (i, j)

    for i, _ in enumerate(intersection_segments):
        if len(intersection_segments[i]) != 2:
            raise ValueError("Should have two intersections per id")

        # Get intersection data per segment
        first_segment_indices: tuple[int, int] = intersection_segments[i][0]
        second_segment_indices: tuple[int, int] = intersection_segments[i][1]
        first_segment_index: SegmentIndex = first_segment_indices[0]
        second_segment_index: SegmentIndex = second_segment_indices[0]
        first_intersection_index: SegmentIndex = first_segment_indices[1]
        second_intersection_index: SegmentIndex = second_segment_indices[1]

        first_segment_data: IntersectionData
        second_segment_data: IntersectionData
        first_segment_data = intersection_data_ref[first_segment_index][first_intersection_index]
        second_segment_data = intersection_data_ref[second_segment_index][second_intersection_index]

        #  Don't process intersections already marked as redundant
        if intersection_data_ref[first_segment_index][first_intersection_index].is_redundant:
            continue
        if intersection_data_ref[second_segment_index][second_intersection_index].is_redundant:
            continue

        # Remove intersections at endpoints between adjacent segments
        if out_array[to_array[first_segment_index]] == second_segment_index:
            if (first_segment_data.is_tip) or (second_segment_data.is_base):
                (intersection_data_ref[first_segment_index][first_intersection_index]
                 ).is_redundant = True
                (intersection_data_ref[second_segment_index][second_intersection_index]
                 ).is_redundant = True
        if out_array[to_array[second_segment_index]] == first_segment_index:
            if (second_segment_data.is_tip) or (first_segment_data.is_base):
                (intersection_data_ref[first_segment_index][first_intersection_index]
                 ).is_redundant = True
                (intersection_data_ref[second_segment_index][second_intersection_index]
                 ).is_redundant = True

        # If a contour intersects the tip of a contour and there is an intersection
        # with next contour that is not already marked as redundant, mark this one
        # as redundant
        if first_segment_data.is_tip:
            for j, _ in enumerate(intersection_data_ref[second_segment_index]):
                if intersection_data_ref[second_segment_index][j].is_redundant:
                    continue
                third_segment_index: SegmentIndex
                third_segment_index = (intersection_data_ref[second_segment_index][j]
                                       ).intersection_index

                if out_array[to_array[first_segment_index]] == third_segment_index:
                    (intersection_data_ref[first_segment_index][first_intersection_index]
                     ).is_redundant = True
                    (intersection_data_ref[second_segment_index][second_intersection_index]
                     ).is_redundant = True
        if second_segment_data.is_tip:
            for j, _ in enumerate(intersection_data_ref[first_segment_index]):
                if intersection_data_ref[first_segment_index][j].is_redundant:
                    continue
                third_segment_index: SegmentIndex
                third_segment_index = (intersection_data_ref[first_segment_index][j]
                                       ).intersection_index

                if out_array[to_array[second_segment_index]] == third_segment_index:
                    (intersection_data_ref[first_segment_index][first_intersection_index]
                     ).is_redundant = True
                    (intersection_data_ref[second_segment_index][second_intersection_index]
                     ).is_redundant = True


def split_segment_at_knot(original_segment_index: SegmentIndex,
                          knot: float,
                          to_array_ref: list[NodeIndex],
                          out_array_ref: list[SegmentIndex],
                          segments_ref: list[SegmentGeometry],
                          nodes_ref: list[NodeGeometry],
                          ) -> tuple[SegmentIndex, SegmentIndex, NodeIndex]:
    """
    Helper function to split a given segment at a knot, updating the curve
    network data and getting the indices of the new node and segments in them

    FIXME: Make method more Pythonic since it returns values but also modifies parameters
    by reference.

    Right now this modifies 

    :param original_segment_index: [in]
    :param knot: [in]
    :param to_array_ref: [out] to_array modified by reference
    :type to_array_ref: list[NodeIndex]
    :param out_array_ref: [out] out_array modified by reference
    :type out_array_ref: list[SegmentIndex]
    :param segments_ref: [out] segments modified by reference
    :type segments_ref: list[SegmentGeometry]
    :param nodes_ref: [out] nodes modified by reference
    :type nodes_ref: list[NodeGeometry]

    :return: lower_segment_index
    :return: upper_segment_index
    :return: knot_node_index
    """
    # Check segment validity
    if len(to_array_ref) != len(segments_ref):
        raise ValueError("Segment geometry and topology mismatch")

    # Check node validity
    if len(out_array_ref) != len(nodes_ref):
        raise ValueError("Node geometry and topology mismatch")

    # Split the segment at the knot
    lower_segment: SegmentGeometry
    upper_segment: SegmentGeometry
    lower_segment, upper_segment = segments_ref[original_segment_index].split_at_knot(knot)
    logger.info("Segment %s split into %s and %s at %s",
                segments_ref[original_segment_index],
                lower_segment,
                upper_segment,
                knot)

    # Add the new split segments to the array
    new_segment_index: SegmentIndex = len(segments_ref)
    segments_ref[original_segment_index] = lower_segment
    segments_ref.append(upper_segment)
    nodes_ref.append(NodeGeometry())

    # Update output segment indices
    lower_segment_index = original_segment_index
    upper_segment_index = new_segment_index

    # Add node for the knot to the topology
    original_segment_to_index: NodeIndex = to_array_ref[original_segment_index]
    knot_node_index: NodeIndex = len(out_array_ref)
    out_array_ref.append(new_segment_index)
    to_array_ref[original_segment_index] = knot_node_index
    to_array_ref.append(original_segment_to_index)

    return lower_segment_index, upper_segment_index, knot_node_index


def split_segments_at_intersections(intersection_data: list[list[IntersectionData]],
                                    num_intersections: int,
                                    to_array: list[NodeIndex],
                                    out_array: list[SegmentIndex],
                                    segments: list[SegmentGeometry],
                                    nodes: list[NodeGeometry]
                                    ) -> tuple[list[int], list[list[int]], list[list[int]]]:
    # ) -> tuple[list[SegmentIndex],
    #    dict[int, dict[int, SegmentIndex]],
    #    dict[int, dict[int, NodeIndex]]]:
    """
    Split segments at all of the intersection points, and create maps from the
    new segments to their original indices and from the original indices to their
    corresponding split segment indices
    """
    logger.info("Splitting segments at intersections")

    num_segments: int = len(segments)
    num_nodes: int = len(nodes)

    # split_segment_indices: dict[int, dict[int, SegmentIndex]] = defaultdict(dict)
    # intersection_nodes: dict[int, dict[int, NodeIndex]] = defaultdict(dict)
    split_segment_indices: list[list[SegmentIndex]] = [[] for _ in range(num_segments)]
    intersection_nodes: list[list[NodeIndex]] = [[] for _ in range(num_intersections)]

    # Check segment validity
    if len(to_array) != len(segments):
        raise ValueError("Segment geometry and topology mismatch")

    # Check node validity
    if len(out_array) != len(nodes):
        raise ValueError("Node geometry and topology mismatch")

    # Initially the map is the identity
    # FIXME: potentially incompatible C++ translation below
    original_segment_indices: list[SegmentIndex] = arange(num_segments)
    for i in range(num_segments):
        split_segment_indices[i].append(i)

    # Mark all base nodes before modifying connectivity
    from_array: list[NodeIndex]
    from_array = AbstractCurveNetwork.build_from_array(to_array, out_array)
    for segment_index, _ in enumerate(intersection_data):
        # segment: SegmentGeometry = segments[segment_index]
        segment_intersection_data: list[IntersectionData] = intersection_data[segment_index]

        for j, _ in enumerate(segment_intersection_data):
            # Skip redundant information
            if segment_intersection_data[j].is_redundant:
                continue

            # Mark intersections at the tip of the contour
            if segment_intersection_data[j].is_base:
                # Add node to global registry
                intersection_id: int = segment_intersection_data[j].id
                intersection_node_index: NodeIndex = from_array[segment_index]
                intersection_nodes[intersection_id].append(intersection_node_index)

    for i, _ in enumerate(intersection_data):
        # Get segment i and sorted intersections
        segment_index: SegmentIndex = i
        segment_intersection_data: list[IntersectionData] = intersection_data[segment_index]
        # TODO: check that the below sorts by reference.
        segment_intersection_data.sort(key=lambda data: data.knot)

        # split_segment_indices[segment_index]
        for j, _ in enumerate(segment_intersection_data):
            # Skip redundant intersections
            if segment_intersection_data[j].is_redundant:
                continue

            # Skip already handled base points
            if segment_intersection_data[j].is_base:
                continue

            # Mark intersections at the tip of the contour
            if segment_intersection_data[j].is_tip:
                # Mark base as intersection
                intersection_node_index: NodeIndex = to_array[segment_index]

                # Add node to global registry
                intersection_id: int = segment_intersection_data[j].id
                intersection_nodes[intersection_id].append(intersection_node_index)
            else:
                # FIXME (from ASOC) The base is currently not tracked and can't be removed
                # As a hack, length zero intersections are removed
                intersection: float = segment_intersection_data[j].knot
                lower_segment_index: SegmentIndex
                upper_segment_index: SegmentIndex
                intersection_node_index: NodeIndex
                (lower_segment_index,
                 upper_segment_index,
                 intersection_node_index) = split_segment_at_knot(
                    segment_index,
                    intersection,
                    to_array,
                    out_array,
                    segments,
                    nodes)

                # Add node to global registry
                intersection_id = segment_intersection_data[j].id
                intersection_nodes[intersection_id].append(intersection_node_index)

                # Update segment indices record
                original_segment_indices.append(-1)
                split_segment_indices[i][-1] = lower_segment_index
                split_segment_indices[i].append(upper_segment_index)
                original_segment_indices[lower_segment_index] = i
                original_segment_indices[upper_segment_index] = i

                # Continue splitting upper segment
                segment_index = upper_segment_index

    logger.debug("Split %s segments with %s nodes into %s segments with %s nodes",
                 num_segments,
                 num_nodes,
                 len(segments),
                 len(nodes))

    return original_segment_indices, split_segment_indices, intersection_nodes


def connect_segment_intersections(segments: list[SegmentGeometry],
                                  intersection_data: list[list[IntersectionData]],
                                  intersection_nodes: list[list[NodeIndex]],
                                  to_array: list[NodeIndex],
                                  out_array: list[SegmentIndex],
                                  split_segment_indices: list[list[SegmentIndex]],
                                  intersection_array: list[NodeIndex],
                                  nodes: list[NodeGeometry]
                                  ) -> None:
    """
    Create a map from nodes to their corresponding intersection point if they are
    an intersection node and -1 otherwise.

    :param intersection_array_ref: [out]
    :param nodes_ref: [out]
    """
    logger.info("Connecting segments intersections")

    # use_combinatorial_data = True
    # if use_combinatorial_data:
    for i, _ in enumerate(intersection_nodes):
        if len(intersection_nodes[i]) == 0:
            continue
        if len(intersection_nodes[i]) != 2:
            raise ValueError("Node intersection is not a pair")

        first_intersection_node: NodeIndex = intersection_nodes[i][0]
        second_intersection_node: NodeIndex = intersection_nodes[i][1]

        # Ignore intersections that already exist
        if intersection_array[first_intersection_node] >= 0:
            continue
        elif intersection_array[second_intersection_node] >= 0:
            continue
        else:
            intersection_array[first_intersection_node] = second_intersection_node
            intersection_array[second_intersection_node] = first_intersection_node
            nodes[first_intersection_node].mark_as_intersection()
            nodes[second_intersection_node].mark_as_intersection()

    # FIXME: below still had a couple more lines of code that were not doing anything...


def split_segments_at_cusps(interior_cusps: list[list[float]],
                            original_segment_indices_ref: list[SegmentIndex],
                            split_segment_indices_ref: list[list[SegmentIndex]],
                            to_array_ref: list[NodeIndex],
                            out_array_ref: list[SegmentIndex],
                            intersection_array_ref: list[NodeIndex],
                            segments_ref: list[SegmentGeometry],
                            nodes_ref: list[NodeGeometry]):
    """
    Split segments at all of the cusp points, and create maps from the new
    segments to their original indices and from the original indices to their
    corresponding split segment indices
    """
    logger.info("Splitting segments at cusps")
    num_segments: int = len(segments_ref)
    num_nodes: int = len(nodes_ref)
    num_original_segments: int = len(interior_cusps)

    # Check segment validity
    if len(to_array_ref) != len(segments_ref):
        raise ValueError("Segment geometry and topology mismatch")

    # Check node validity
    if len(out_array_ref) != len(nodes_ref):
        raise ValueError("Node geometry and topology mismatch")

    for i in range(num_original_segments):
        for j, _ in enumerate(interior_cusps[i]):
            cusp: float = interior_cusps[i][j]
            for k, _ in enumerate(split_segment_indices_ref[i]):
                segment_partition_index: SegmentIndex = split_segment_indices_ref[i][k]
                segment_partition: SegmentGeometry = segments_ref[segment_partition_index]

                if segment_partition.parameter_curve.domain.is_in_interior(cusp):
                    lower_segment_index: SegmentIndex
                    upper_segment_index: SegmentIndex
                    knot_node_index: NodeIndex
                    (lower_segment_index,
                     upper_segment_index,
                     knot_node_index) = split_segment_at_knot(segment_partition_index,
                                                              cusp,
                                                              to_array_ref,
                                                              out_array_ref,
                                                              segments_ref,
                                                              nodes_ref)
                    # Mark new node as interior cusp
                    logger.info("Marking node %s as interior cusp", knot_node_index)
                    nodes_ref[knot_node_index].mark_as_interior_cusp()

                    # Add trivial intersection information
                    num_nodes = len(intersection_array_ref)
                    if knot_node_index == num_nodes:
                        intersection_array_ref.append(-1)
                    else:
                        raise ValueError("Invalid node index used for knot split")

                    # Update segment indices record
                    split_segment_indices_ref[i][k] = lower_segment_index
                    split_segment_indices_ref[i].append(upper_segment_index)

                    assert ((lower_segment_index == len(original_segment_indices_ref))
                            or (upper_segment_index == len(original_segment_indices_ref)))
                    original_segment_indices_ref.append(i)  # FIXME (ASOC): Risky assumption
                    original_segment_indices_ref[lower_segment_index] = i
                    original_segment_indices_ref[upper_segment_index] = i
                    break

    logger.debug("Split %s segments with %s nodes into %s segments with %s nodes",
                 num_segments,
                 num_nodes,
                 len(segments_ref),
                 len(nodes_ref))
