from enum import Enum


class ConicType(Enum):
    ELLIPSE = 1
    HYPERBOLA = 2
    PARABOLA = 3
    PARALLEL_LINES = 4
    INTERSECTING_LINES = 5
    LINE = 6
    POINT = 7
    EMPTY = 8
    PLANE = 9
    ERROR = 10
    UNKNOWN = 11


class Conic(RationalFunction):
    def __init__(self):

    @classmethod
