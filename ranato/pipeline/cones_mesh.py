"""
Optional executable for locating cone vertices within a mesh.
"""


from abc import ABC


class ConeLocator(ABC):
    """ Abstract class providing structure for any external algorithm locating cone vertices.

    Args:
        ABC (_type_): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
