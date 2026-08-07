"""
UI layer.
"""

import bpy
from bpy.types import Operator, PointerProperty, PropertyGroup

from ...common import ADDON_ID
from .bff import BFFSettings, BFFStrategy
from .campen import CampenSettings, CampenStrategy
from .ceps import CEPSSettings, CEPSStrategy
from .cetm import CETMSettings, CETMStrategy

# TODO: separate blender-facing files and the regular files...
# TODO: Implement future version utilize vertex position, texture coordinates, and face indices
#       of the mesh directly from Blender rather than an .obj file.

STRATEGIES: dict[str, CampenStrategy | CEPSStrategy | CETMSettings | BFFStrategy] = {
    strategy.bl_idname: strategy() for strategy in (CampenStrategy, CEPSStrategy, CETMStrategy, BFFStrategy)
}


class UVUnwrapperSettings(PropertyGroup):

    # NOTE: method for use in detecting which settings to display in the UI panel
    method: bpy.props.EnumProperty(
        name="UV Unwrapper",
        description="Choose preferred UV unwrapping algorithm",
        items=[
            ("CAMPEN", "campen", ""),
            ("CEPS", "ceps", ""),
            ("CETM", "cetm", ""),
            ("BFF", "bff", ""),
        ],
        default="CAMPEN"
    )

    campen: bpy.props.PointerProperty(type=CampenSettings)
    ceps: bpy.props.PointerProperty(type=CEPSSettings)
    cetm: bpy.props.PointerProperty(type=CETMSettings)
    bff: bpy.props.PointerProperty(type=BFFSettings)


class RANATO_OT_uv_unwrap(Operator):
    """
    Brings up UI panel for searching for a particular mesh. 
    Then, returns the string key for the particular mesh. 

    For context:
    https://blender.stackexchange.com/questions/19416/what-do-operator-methods-do-poll-invoke-execute-draw-modal
    """
    bl_idname = "object.uv_unwrap"
    bl_label = "UV Unwrap"

    # https://docs.blender.org/api/current/bpy.types.Depsgraph.html
    def execute(self, context: bpy.types.Context) -> set:
        """
        Execute the operator.
        Grabs the objects within the dependency graph.
        """

        self.report({'INFO'}, "Calling UV unwrapper...")

        # --- Select the strategy from a list of strategies or something... ---
        # obj = context.active_object
        settings: PointerProperty = context.scene.uv_unwrap_settings
        strategy: CampenStrategy | CEPSStrategy | CETMSettings | BFFStrategy = STRATEGIES[
            settings.method]
        self.report({"INFO"}, f"Using {settings.method} UV unwrapping")
        strategy.execute(context, settings)

        # TODO: need to perform checks to see if the UV unwrapping is... well, valid!
        # CEPS is quite finicky with its UV unwrappings from what I've seen...
        # Sometimes CAMPEN UV unwrapping outputs nothing or a blank file, so be sure to check for that
        self.report({'INFO'}, message="Generated UV unwrapping!")
        return {'FINISHED'}
