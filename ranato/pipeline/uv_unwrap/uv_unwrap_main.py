"""
UI layer.
"""

import pathlib

import bpy
import bpy.types

from .bff import BFFStrategy
from .campen import CampenStrategy
from .ceps import CEPSStrategy
from .cetm import CETMStrategy
from .uv_unwrap_settings import UVUnwrapperSelection

# TODO: separate blender-facing files and the regular files...
# TODO: Implement future version utilize vertex position, texture coordinates, and face indices
#       of the mesh directly from Blender rather than an .obj file.

STRATEGIES: dict[str, CampenStrategy | CEPSStrategy | CETMStrategy | BFFStrategy] = {
    strategy.bl_idname: strategy() for strategy in (CampenStrategy, CEPSStrategy, CETMStrategy, BFFStrategy)
}


def validate_uv_unwrapping(filepath: pathlib.Path) -> bool:
    """ Checks to see if UV unwrapping .obj output is empty or not.

    :param filepath: path pointing to .obj file
    :type filepath: pathlib.Path
    :return: true if valid (not empty), else false
    :rtype: bool
    """
    return filepath.stat().st_size != 0


class RANATO_OT_uv_unwrap(bpy.types.Operator):
    """
    Calls UV unwrapper script.

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
        settings: UVUnwrapperSelection = context.scene.uv_unwrap_settings
        strategy: CampenStrategy | CEPSStrategy | CETMStrategy | BFFStrategy = STRATEGIES[
            settings.method]
        self.report({"INFO"}, f"Using {settings.method} UV unwrapping")
        strategy.execute(context, settings)

        # TODO: if the UV unwrapping is actually finished, check the output file to see that it's not empty.

        # TODO: have option to import UV unwrapping into Blender + swap to UV unwrapping view to
        # see UV unwrapping visualized!
        # bpy.ops.screen.userpref_show('INVOKE_DEFAULT')
        # new_window = bpy.context.window_manager.windows[-1]
        # popup_area = new_window.screen.areas[0]
        # popup_area.type = 'IMAGE_EDITOR'
        # popup_area.ui_type = 'UV'

        # TODO: need to perform checks to see if the UV unwrapping is... well, valid!
        # CEPS is quite finicky with its UV unwrappings from what I've seen...
        # Sometimes CAMPEN UV unwrapping outputs nothing or a blank file, so be sure to check for that
        self.report({'INFO'}, message="Generated UV unwrapping!")
        return {'FINISHED'}
