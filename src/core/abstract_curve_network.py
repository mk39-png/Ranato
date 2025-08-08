"""
Methods to build an curve network from minimal connectivity data.
Used for contour_network.
"""

import logging

from src.core.common import logger, vector_contains

# Typedefs for readability.
NodeIndex = int
SegmentIndex = int


def is_valid_curve_data(to_array: list[NodeIndex],
                        out_array: list[SegmentIndex]) -> bool:
    """
    Check if input has valid indexing, meaning all to nodes are valid
    Note that out may be invalid for some nodes if they are terminal
    @param[in] to_array: array mapping segments to their endpoints
    @param[in] out_array: array mapping nodes to their outgoing segment
    @return true iff the curve data is valid
    """
    num_segments: int = len(to_array)
    num_nodes: int = len(out_array)

    for si in range(num_segments):
        if (to_array[si] < 0) or to_array[si] >= num_nodes:
            logger.error("Segment %s is invalid with to node %s", si, to_array[si])

    return True


def is_valid_minimal_curve_network_data(to_array: list[NodeIndex],
                                        out_array: list[SegmentIndex],
                                        intersection_array: list[NodeIndex]) -> bool:
    """
    Check if input describes a valid curve network
    @param[in] to_array: array mapping segments to their endpoints
    @param[in] out_array: array mapping nodes to their outgoing segment
    @param[in] intersection_array: list of intersection nodes
    @return true iff the curve network data is valid
    """
    num_segments: int = len(to_array)
    num_nodes: int = len(out_array)

    if len(to_array) != num_segments:
        logger.error("to domain not in bijection with number of segments")
        return False
    if len(out_array) != num_nodes:
        logger.error("out domain not in bijection with number of nodes")
        return False
    if len(intersection_array) != num_nodes:
        logger.error("out domain not in bijection with number of nodes")
        return False

    # Check all out nodes are valid (to and intersection array can have invalid
    # nodes)
    if not is_valid_curve_data(to_array, out_array):
        return False

    return True


class AbstractCurveNetwork():
    """
    An abstract curve network is a graph representing a finite set of possibly
    intersecting directed curves. For simplicity, it is assumed that all
    intersections are either transversal or T-nodes so that at most two curves
    intersect at a node.
    """

    def __init__(self,
                 to_array: list[NodeIndex],
                 out_array: list[SegmentIndex],
                 intersection_array: list[NodeIndex]) -> None:
        """
        Construct the network from the basic topological information.
        @param[in] to_array: array mapping segments to their endpoints
        @param[in] out_array: array mapping nodes to their outgoing segment
        @param[in] intersection_array: list of intersection nodes
        """
        self.__to_array: list[NodeIndex] = to_array
        self.__out_array: list[SegmentIndex] = out_array
        self.__intersection_array: list[NodeIndex] = intersection_array

        if logger.getEffectiveLevel() == logging.DEBUG:
            if not is_valid_minimal_curve_network_data(to_array, out_array, intersection_array):
                raise ValueError("Could not build abstract curve network")
                # Rather than raising a ValueError, could perhaps log the error and then catch the error on the way back?
                # That way, the program doesn't crash, kind of.
                # TODO: clear topology?

        # Build curve network
        init_abstract_curve_network()

        # Check validity
        if logger.getEffectiveLevel() == logging.DEBUG:
            if not self.__is_valid_abstract_curve_network():
                raise ValueError("Inconsistent abstract curve network built")

    @property
    def num_segments(self):
        """
        Return the number of segments in the curve network.
        @return number of segments
        """
        return self.__num_segments

    @property
    def num_nodes(self):
        """
        Return the number of nodes in the curve network.
        @return number of nodes
        """
        return self.__num_nodes

    def next(self, segment_index: SegmentIndex) -> SegmentIndex:
        """
        Get the next segment after a given segment (or -1 if there is no next
        segment)
        :param[in] segment_index: query segment index
        :return next segment
        """
        if not self._is_valid_segment_index(segment_index):
            return -1
        return self.next_array[segment_index]

    def prev(self, segment_index: SegmentIndex) -> SegmentIndex:
        """
        Get the previous segment after a given segment (or -1 if there is no
        previous segment)
        @param[in] segment_index: query segment index
        @return previous segment
        """
        if not self._is_valid_segment_index(segment_index):
            return -1
        return self.prev_array[segment_index]

    def to(self, segment_index: SegmentIndex) -> NodeIndex:
        """
        Get the node at the tip of the segment
        Note that this operation is valid for any valid segment
        @param[in] segment_index: query segment index
        @return to node of the segment
        """
        if not self._is_valid_segment_index(segment_index):
            return -1
        return self.to_array[segment_index]

    def from_(self, segment_index: SegmentIndex) -> NodeIndex:
        """
        Get the node at the base of the segment
        Note that this operation is valid for any valid segment
        @param[in] segment_index: query segment index
        @return from node of the segment
        """
        if not self._is_valid_segment_index(segment_index):
            return -1
        return self.from_array[segment_index]

    def intersection(self, node_index: NodeIndex) -> NodeIndex:
        """
        Get the node that intersects the given node (or -1 if the node does not
        intersect another node)
        @param[in] node_index: query node index
        @return intersection node of the node
        """
        if not self._is_valid_node_index(node_index):
            return -1
        return self.intersection_array[node_index]

    def out(self, node_index: NodeIndex) -> SegmentIndex:
        """
        Get the outgoing segment for the node (or -1 if none exists)
        @param[in] node_index: query node index
        @return out segment of the node
        """
        if not self._is_valid_node_index(node_index):
            return -1
        return self.out_array[node_index]

    def in_(self, node_index: NodeIndex) -> SegmentIndex:
        """
        Get the incoming segment for the node (or -1 if none exists)
        @param[in] node_index: query node index
        @return in segment of the node
        """
        if not self._is_valid_node_index(node_index):
            return -1
        return self.in_array[node_index]

    def is_boundary_node(self, node_index: NodeIndex) -> bool:
        """
        Determine if the node is on the boundary of a curve in the curve network.
        @param[in] node_index: query node index
        @return true iff the given node is a boundary node
        """

    def has_intersection_node(self, node_index: NodeIndex) -> bool:
        """
        Determine if the node has an intersection.
        @param[in] node_index: query node index
        @return true iff the given node is an intersection node
        """

    def is_tnode(self, node_index: NodeIndex) -> bool:
        """
        Determine if the node is a T-node, i.e., has an intersection and one of
        the two is on the boundary.
        Note that this is a weaker condition than having an intersection node and
        being and intersection node and is not simply a logical and of the two
        conditions.
        @param[in] node_index: query node index
        @return true iff the given node is a boundary intersection node
        """

    # ***********
    # Getters
    # ***********

    @property
    def next_array(self):
        return self.__next_array

    @property
    def prev_array(self):
        return self.__prev_array

    @property
    def to_array(self) -> list[int]:
        return self.__to_array

    @property
    def from_array(self):
        return self.__from_array

    @property
    def intersection_array(self) -> list[int]:
        return self.__intersection_array

    @property
    def out_array(self) -> list[int]:
        return self.__out_array

    @property
    def in_array(self):
        return self.__in_array

    # ******************
    #  Public methods
    # ******************

    @staticmethod
    def build_from_array(to_array: list[NodeIndex],
                         out_array: list[SegmentIndex]) -> list[NodeIndex]:
        """
        Build from map sending segments to their origin nodes.
        @param[in] to_array: array mapping segments to their endpoints
        @param[in] out_array: array mapping nodes to their outgoing segment
        @param[out] from_array: array mapping segments to their origin points
        """

    # ******************
    #  Protected methods
    # ******************
    def _is_valid_segment_index(self, segment_index: SegmentIndex) -> bool:
        """
        Determine if the index describes a segment of the curve network
        """
        # Ensure in bounds for segment list
        if segment_index < 0:
            return False
        if segment_index >= self.num_segments:
            return False
        return True

    def _is_valid_node_index(self, node_index: NodeIndex) -> bool:
        """
        Determine if the index describes a node of the curve network
        """
        # Ensure in bounds for node list
        if node_index < 0:
            return False
        if node_index >= self.num_nodes:
            return False
        return True

    # ****************
    #  Private methods
    # ****************

    def __is_valid_abstract_curve_network(self) -> bool:
        """
        General validity checker for the network topology
        """
        num_segments: int = self.num_segments
        num_nodes: int = self.num_nodes

        # Array size checks
        if len(self.next_array) != num_segments:
            logger.error("Inconsistent next array")
            return False
        if len(self.prev_array) != num_segments:
            logger.error("Inconsistent prev array")
            return False

        if len(self.to_array) != num_segments:
            logger.error("Inconsistent to array")
            return False

        if len(self.from_array) != num_segments:
            logger.error("Inconsistent from array")
            return False

        if len(self.intersection_array) != num_nodes:
            logger.error("Inconsistent intersection array")
            return False

        if len(self.out_array) != num_nodes:
            logger.error("Inconsistent out array")
            return False

        if len(self.in_array) != num_nodes:
            logger.error("Inconsistent in array")
            return False

        # Check segment topology
        for si in range(self.num_segments):
            #  Check to node
            if not self._is_valid_node_index(self.to(si)):
                logger.error("To does not have a valid endpoint for segment %s", si)
                return False
            if self.in_(self.to(si)) != si:
                logger.error("in(to(s)) is not the identity for segment %s", si)
                return False

            # Check from node
            if not self._is_valid_node_index(self.from_(si)):
                logger.error("From does not have a valid endpoint for segment %s", si)
                return False
            if self.out(self.from_(si)) != si:
                logger.error("out(from(s)) is not the identity for segment %s", si)
                return False

            # Check next segment is consistent if it exists
            if self._is_valid_segment_index(self.next(si)):
                if self.prev(self.next(si)) != si:
                    logger.error(
                        "prev(next(s)) is not the identity for nonterminal segment %s", si)
                    logger.error("next(s) is %s", self.next(si))
                    return False

            #  Check to node is an endpoint if the next segment does not exist
            else:
                if self._is_valid_segment_index(self.out(self.to(si))):
                    logger.error("Terminal segment %s does not have a terminal endpoint", si)
                    return False

            # Check prev segment is consistent if it exists
            if self._is_valid_segment_index(self.prev(si)):
                if self.next(self.prev(si)) != si:
                    logger.error(
                        "next(prev(s)) is not the identity for noninitial segment %s", si)
                    return False

            # Check to node is an endpoint if the next segment does not exist
            else:
                if self._is_valid_segment_index(self.in_(self.from_(si))):
                    logger.error("Initial segment %s does not have a initial start point", si)
                    return False

        # Check node topology
        is_out_segment: list[bool] = [False] * self.num_segments
        is_in_segment: list[bool] = [False] * self.num_segments
        for ni in range(self.num_nodes):
            # Check the outgoing segment comes from the node if it exists
            if self._is_valid_segment_index(self.out(self.in_(ni))):
                if self.from_(self.out(ni)) != ni:
                    logger.error(
                        "from(out(n)) is not the identity for nonterminal node %s", ni)
                    return False
                is_out_segment[self.out(ni)] = True

            # Check the incoming segment goes to the node if it exists
            if self._is_valid_segment_index(self.in_(ni)):
                if self.to(self.in_(ni)) != ni:
                    logger.error("to(in(n)) is not the identity for non initial node %s",
                                 ni)
                    return False
                is_in_segment[self.in_(ni)] = True

            # Check the intersection is a closed order 2 loop if it exists
            if self._is_valid_node_index(self.intersection(ni)):
                if self.intersection(self.intersection(ni)) != ni:
                    logger.error("Intersection is order 2 for intersection node %s", ni)
                    return False

        # Check all segments originate from some node
        if vector_contains(is_out_segment, False):
            logger.error("Segment does not have a starting node")
            return False

        # Check all segments go into some node
        if vector_contains(is_in_segment, False):
            logger.error("Segment does not have a terminal node")
            return False

        return True
