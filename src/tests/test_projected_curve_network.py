"""
Projected Curve Network testing
"""

from src.core.abstract_curve_network import AbstractCurveNetwork
from src.core.common import NodeIndex, SegmentIndex

# ************************
# Main Test Cases
# ************************


# ************************
# Original C++ test cases
# ************************

def test_closed_loop() -> None:
    """
    "A curve network can be built from topology information", "[projected_curve_network]"
    """
    out_array: list[SegmentIndex] = [0, 1, 2]
    to_array: list[NodeIndex] = [1, 2, 0]
    intersections: list[NodeIndex] = [-1, -1, -1]
    curve_network = AbstractCurveNetwork(to_array, out_array, intersections)

    assert curve_network.next(0) == 1
    assert curve_network.next(1) == 2
    assert curve_network.next(2) == 0
    assert curve_network.prev(0) == 2
    assert curve_network.prev(1) == 0
    assert curve_network.prev(2) == 1
    assert curve_network.to(0) == 1
    assert curve_network.to(1) == 2
    assert curve_network.to(2) == 0
    assert curve_network.from_(0) == 0
    assert curve_network.from_(1) == 1
    assert curve_network.from_(2) == 2
    assert curve_network.out(0) == 0
    assert curve_network.out(1) == 1
    assert curve_network.out(2) == 2
    assert curve_network.in_(0) == 2
    assert curve_network.in_(1) == 0
    assert curve_network.in_(2) == 1
