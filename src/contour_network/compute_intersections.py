"""
Methods to compute intersections for quadratic surfaces.
"""

from dataclasses import dataclass


@dataclass
class IntersectionParameters():
    """
    Parameters for intersection computations.
    """
    # If true, use heuristics to check if there are no intersections
    use_heuristics: bool = True
    # Amount to trim ends of contour segments by; intersections that are trimmed are clamped
    # to the endpoint.
    trim_amount: float = 1e-5
