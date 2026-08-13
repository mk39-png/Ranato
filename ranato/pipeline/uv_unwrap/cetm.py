import pathlib

import bpy
from bpy.types import PropertyGroup
from confmap.confmap import CETM
from confmap.io_utils import read_obj, write_obj
from confmap.mesh_utils import TriangleMesh

from ...common import ADDON_ID, INPUT_OBJ_FILENAME, OUTPUT_OBJ_FILENAME
from .uv_unwrap_strategy import UVUnwrapStrategy


class CETMSettings(PropertyGroup):
    """
    Settings for CETM conformal mapping method (if there are any settings in the future)
    """


class CETMStrategy(UVUnwrapStrategy):
    """ Implementation of CETM UV Unwrapping strategy.

    Args:
        UVUnwrapStrategy (_type_): _description_
    """
    _id = "cetm"  # used for accessing attributes list and whatnot
    bl_idname = "CETM"  # TODO: rename to something more descriptive, like strategy.cetm... maybe
    bl_label = "CETM"

    def _call_uv_unwrapper(self, args=None) -> None:

        directory_temp: pathlib.Path = pathlib.Path(
            bpy.context.preferences.addons[ADDON_ID].preferences.directory_temp)
        filepath_input: str = (directory_temp / INPUT_OBJ_FILENAME).as_posix()
        filepath_output: str = (directory_temp / OUTPUT_OBJ_FILENAME).as_posix()

        vertices, faces = read_obj(filepath_input)

        # TODO: allow spot for inputting boundary vertices for non-topological spheres
        conformal_map = CETM(vertices, faces)
        uv_unwrapping_image: TriangleMesh = conformal_map.layout()

        write_obj(filepath_output,
                  conformal_map.vertices, conformal_map.faces, uv_unwrapping_image.vertices, uv_unwrapping_image.faces)

    def execute(self, context, settings) -> None:
        """ Calls helper function for UV unwrapping and performs validation if needed.

        Args:
            context (_type_): _description_
            settings (_type_): _description_
        """
        self._call_uv_unwrapper()
