# Setting up addon with multiple files
# https://blender.stackexchange.com/questions/202570/multi-files-to-addon

# Basic panel setup
# https://docs.blender.org/api/current/bpy.types.Panel.html

# Referencing below to see how Render Engine interacts with Panel
# (and changing it all so that the panel only appears when user activates Ranato)
# https://github.com/bnpr/Malt/blob/725f509ab25be736cb592cf1e9d5258ed4271e8a/BlenderMalt/MaltMaterial.py#L98
# https://github.com/Griperis/BlenderDataVis/blob/master/data_vis/operators/surface_chart.py


import bpy
from bpy.types import AddonPreferences

from ...common import ADDON_ID, INPUT_OBJ_FILENAME
from .uv_unwrap_strategy import UVUnwrapStrategy


class CampenStrategy(UVUnwrapStrategy):
    _id = "campen"
    bl_idname = "CAMPEN"
    bl_label = "Campen et al. 2021"

    def execute(self, context, settings) -> None:
        """
        # TODO: refactor so that function is universal for any UV unwrapper
        # TODO: this means passing in the args and params and whatnot...

        Calls the uv_unwrapper algorithm with specified commands.
        Also calls to activate the Conda environment if that has not been done already.
        """
        # Call sys subprocess to execute python script in another file.
        preferences: AddonPreferences | None = bpy.context.preferences.addons[
            ADDON_ID].preferences
        filepath_conda: str = preferences.filepath_conda
        file_path_uv_unwrapper: str = preferences.filepath_uv_unwrap_campen
        directory_temp: str = preferences.directory_temp

        self._retrieve_vertex_angles(context)

        # TODO: allow user to specify parameters into this... by writing down whatever in the panel box
        script_args: list[str] = [
            filepath_conda,
            file_path_uv_unwrapper,
            "--input", directory_temp,
            "--fname", INPUT_OBJ_FILENAME,
            "--output", directory_temp,
            "--output_type", "param",
            "--output_format", "obj",
            "--error_log"
        ]
        user_args: list[str] = self._process_properties(settings)
        script_args.extend(user_args)
        self._call_uv_unwrapper(script_args)

        # TODO: perform verification on whether OBJ output is VALID!!!
        # Which is to say... check if temp_out.obj is empty file or not
