import pathlib

import bpy
from bpy.types import UILayout
from confmap.confmap import BFF
from confmap.io_utils import read_obj, write_obj
from confmap.mesh_utils import TriangleMesh

from ...common import ADDON_ID, INPUT_OBJ_FILENAME, OUTPUT_OBJ_FILENAME
from .uv_unwrap_settings import UVUnwrapperSelection
from .uv_unwrap_strategy import UVUnwrapStrategy


class BFFStrategy(UVUnwrapStrategy):
    """_summary_

    Args:
        UVUnwrapStrategy (_type_): _description_
    """
    _id = "bff"
    bl_idname = "BFF"
    bl_label = "BFF"

    def _call_uv_unwrapper(self, args=None) -> None:
        directory_temp: pathlib.Path = pathlib.Path(
            bpy.context.preferences.addons[ADDON_ID].preferences.directory_temp)
        filepath_input: str = (directory_temp / INPUT_OBJ_FILENAME).as_posix()
        filepath_output: str = (directory_temp / OUTPUT_OBJ_FILENAME).as_posix()

        vertices, faces = read_obj(filepath_input)
        conformal_map = BFF(vertices, faces)
        uv_unwrapping_image: TriangleMesh = conformal_map.layout()

        write_obj(filepath_output,
                  conformal_map.vertices, conformal_map.faces, uv_unwrapping_image.vertices, uv_unwrapping_image.faces)

    def draw(self, layout: UILayout, settings: UVUnwrapperSelection) -> None:
        layout.label(text="No adjustable settings exist for BFF algorithm")

    def execute(self, context, settings) -> None:
        self._call_uv_unwrapper()
