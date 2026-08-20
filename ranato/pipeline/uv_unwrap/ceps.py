
import pathlib

import bpy
from bpy.types import AddonPreferences

from ...common import ADDON_ID, INPUT_OBJ_FILENAME, OUTPUT_OBJ_FILENAME
from .uv_unwrap_strategy import UVUnwrapStrategy


class CEPSStrategy(UVUnwrapStrategy):
    _id = "ceps"
    bl_idname = "CEPS"  # TODO: rename ID name?
    bl_label = "CEPS"

    def execute(self, context, settings) -> None:
        # NOTE: this calls an EXE, which is perfect for us!

        # TODO: make helper function that retreives preference?????
        preferences: AddonPreferences | None = bpy.context.preferences.addons[
            ADDON_ID].preferences
        filepath_uv_unwrap_ceps: str = preferences.filepath_uv_unwrap_ceps
        directory_temp: pathlib.Path = pathlib.Path(preferences.directory_temp)
        mesh_filepath: str = (directory_temp / INPUT_OBJ_FILENAME).as_posix()

        # TODO: implement the curvatures retrieval for CEPs
        # retrieve_cone_vertices(context.scene.vertex_angles)
        # self._retrieve_vertex_angles(context)

        script_args: list[str] = [
            filepath_uv_unwrap_ceps,  # script filepath
            mesh_filepath,  # target mesh
            f"--outputLinearTextureFilename={directory_temp / OUTPUT_OBJ_FILENAME}",
            f"--outputLogFilename={directory_temp / 'ceps.log'}"
        ]

        user_args: list[str] = self._process_properties(settings, ceps_format=True)
        script_args.extend(user_args)

        self._call_uv_unwrapper(script_args)
