from bpy.props import BoolProperty, IntProperty
from bpy.types import PropertyGroup

from .uv_unwrap_strategy import UVUnwrapStrategy


class CEPSSettings(PropertyGroup):
    """
    Settings for CEPS algorithm
    """
    # curvatures file
    # scale factors file
    # TODO: may be worth implementing MPZ as another cone_mesh algorithm...
    # For identifying cones and whatnot... making that modular and expandable
    # ffield... from MPZ style crossfield

    # NOTE: descriptions from original repo
    # https://github.com/MarkGillespie/CEPS
    prop_greedyConesMax: IntProperty(
        name="Greedy Cones Maximum",
        description="Maximum allowed log scale factor when placing cones (lower value = lower distortion in final parameterization, default value=5)",
        default=5,
        min=1
    )
    prop_useExactCones: BoolProperty(
        name="Use Exact Cones",
        description="Do not lump together nearby cones in the ffield input, if any. Cones prescribed via --curvatures or --scaleFactors are never lumped",
        default=False
    )
    prop_noFreeBoundary: BoolProperty(
        name="No Free Boundary",
        description="Do not impose minimal-area-distortion boundary conditions (useful, e.g. if prescribing polygonal boundary conditions i.e. specifying angles at vertices)",
        default=False
    )
    prop_viz: BoolProperty(
        # TODO: fix name making it more clear what this is doing
        name="Enable Polyscope GUI (external popup)",
        description="",
        default=False
    )


class CEPSStrategy(UVUnwrapStrategy):
    _id = "ceps"
    bl_idname = "CEPS"  # TODO: rename ID name
    bl_label = "CEPS"

    def _call_uv_unwrapper(self):
        """
        """

    def execute(self, context, settings):
        """_summary_
        """
        raise NotImplementedError("CEPS not yet implemented!")
