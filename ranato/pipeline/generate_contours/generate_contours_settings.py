# pyright: reportInvalidTypeForm = false

# Make the list of settings  to select.

# Basically, need a bunch of dropdowns for EVERYTHING PYAC has for input.
# Menaing

import bpy.props
# Basically, need to turn these int oregisterable Blender enum box things.
# Operators
import bpy.types
from pyalgcon.contour_network.compute_intersections import \
    IntersectionParameters
from pyalgcon.contour_network.contour_network import (InvisibilityMethod,
                                                      InvisibilityParameters)
from pyalgcon.quadratic_spline_surface.optimize_spline_surface import \
    OptimizationParameters
from pyalgcon.utils.projected_curve_networks_utils import SVGOutputMode


class OptimizationSettings(bpy.types.PropertyGroup):
    """ 
    Settings for PYAC smooth surface optimization 
    """
    # -- Main optimization weight options --
    prop_position_difference_factor: bpy.props.FloatProperty(
        name="Position Difference Factor",
        default=1.0, description="main optimization weight option")

    prop_parametrized_quadratic_surface_mapping_factor: bpy.props.FloatProperty(
        name="Parametrized Quadratic Surface Mapping Factor",
        default=1.0, description="main optimization weight option")

    # -- Weights for cone positions, normals, and gradients --
    prop_cone_position_difference_factor: bpy.props.FloatProperty(
        name="Cone Position Difference Factor",
        default=1.0, description="fitting weight for cone vertices")

    prop_cone_vertex_gradient_difference_factor: bpy.props.FloatProperty(
        name="Cone Vertex Gradient Difference Factor",
        default=1e6, description="fitting weight for cone vertex gradients")

    prop_cone_adjacent_position_difference_factor: bpy.props.FloatProperty(
        name="Cone Adjacent Position Difference Factor",
        default=1.0, description="fitting weight for vertices collapsed to a cone")

    prop_cone_adjacent_vertex_gradient_difference_factor: bpy.props.FloatProperty(
        name="Cone Adjacent Vertex Gradient Different Factor",
        default=0.0, description="fitting weight for vertex gradients collapsing to a cone")

    prop_cone_adjacent_edge_gradient_difference_factor: bpy.props.FloatProperty(
        name="Cone Adjacent Edge Gradient Difference Factor",
        default=0.0, description="fitting weight for vertex edge gradients collapsing to a cone")

    prop_cone_normal_orthogonality_factor: bpy.props.FloatProperty(
        name="Cone Normal Orthogonality Factor",
        default=0.0, description="weight for encouraging orthogonality with a normal at a cone")


class IntersectionSettings(bpy.types.PropertyGroup):
    """ 
    Settings for PYAC contour intersections.
    """
    prop_use_heuristics: bpy.props.BoolProperty(
        name="Use Heuristics",
        default=True, description="If true, use heuristics to check if there are no intersections")
    prop_trim_amount: bpy.props.FloatProperty(
        name="Trim Amount",
        default=1e-5, description="Amount to trim ends of contour segments by intersections that are trimmed are clamped to the endpoint.")


class InvisibilitySettings(bpy.types.PropertyGroup):
    """ 
    Settings for PYAC contour invisibility calculation 
    """
    prop_pad_amount: bpy.props.FloatProperty(
        name="Pad Amount",
        default=1e-9, description="Padding for contour domains")

    prop_write_contour_soup: bpy.props.BoolProperty(
        name="Write Contour Soup",
        default=False, description="Option to write contours before graph construction for diagnostics")

    # NOTE: items must maintain ordering as they appear in InvisibilityMethod()
    prop_invisibility_method: bpy.props.EnumProperty(
        name="Invisbility Method",
        items=[("NONE", "NONE",  "Set all QI to 0"),
               ("DIRECT", "DIRECT",  "Ray test per segment"),
               ("CHAINING", "CHAINING",   "Ray test per chain of segments between features"),
               ("PROPAGATION", "PROPAGATION",   "Ray test for connected components with local propagation"),
               ],
        default="CHAINING",
        description="Method for computing quantitative visibility"
    )

    prop_view_intersections: bpy.props.BoolProperty(
        name="View Intersections",
        default=False, description="Options to view each local propagation step during computation for debugging")
    prop_view_cusps: bpy.props.BoolProperty(
        name="View Cusps",
        default=False, description="Options to view each local propagation step during computation for debugging")

    # Options for redundancy checks
    prop_poll_chain_segments: bpy.props.BoolProperty(
        name="Poll Chain Segments",
        default=True, description="Sample and poll 3 segments for majority per chain QI")
    prop_poll_segment_points: bpy.props.BoolProperty(
        name="Poll Segment Points",
        default=False, description="Sample and poll 3 points for majority per segment QI")

    # Consistency checks
    prop_check_chaining: bpy.props.BoolProperty(name="Check Chaining", default=False)
    prop_check_propagation: bpy.props.BoolProperty(name="Check Propagation", default=False)


# def convert_rna_to_pyac_settings(settings: bpy.types.PropertyGroup, dataclass: type):
#     """
#     Converts Blender Property Groups into given Data Class or Enum
#     """

#     # Create dict of properties to match dataclass args
#     properties: list[str] = [
#         name for name in dir(settings) if name.startswith("prop")
#     ]
#     for prop in properties:
#         arg: str = prop.replace("prop_", "")
#         value: float | bool | int = getattr(uv_setting, prop)
#         user_arg: tuple = self._process_single_property(arg, value)

#     args: list = []
#     ci = IntersectionParameters(**args)

#     # uv_setting: CampenSettings | CEPSSettings | BFFSettings | CETMSettings = getattr(
#     #     settings, self._id)

#     # user_args: list[str] = []


# def convert_rna_to_pyac_enum(enum_val: int, enum: type):
#     """

#     """
#     return enum(enum_val)
#     # Need to get context.scene to get the value, which is usually a number
