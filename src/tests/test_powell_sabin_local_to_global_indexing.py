import pytest
from src.quadratic_spline_surface.powell_sabin_local_to_global_indexing import *


def test_build_variable_edge_indices_map() -> None:
    # Making sure that the global_edge_indices intialization is of the correct size and elements.
    num_faces: int = 4
    PLACEHOLDER_INT = -1
    global_edge_indices: list[list[int]] = [[PLACEHOLDER_INT, PLACEHOLDER_INT, PLACEHOLDER_INT]
                                            for _ in range(num_faces)]

    assert len(global_edge_indices) == 4
    assert len(global_edge_indices[0]) == 3
    assert len(global_edge_indices[1]) == 3
    assert len(global_edge_indices[2]) == 3
    assert len(global_edge_indices[3]) == 3

    for _, index in enumerate(global_edge_indices[0]):
        assert index == -1


def test_generate_twelve_split_local_to_global_map():
    """
    Testing to see that the list resizing works as intended
    """
    six_split_local_to_global_map: list[int] = [-1 for _ in range(27)]
    local_to_global_map: list[int] = [39 for _ in range(36)]
    local_to_global_map[0:len(six_split_local_to_global_map)] = six_split_local_to_global_map

    print(local_to_global_map)
