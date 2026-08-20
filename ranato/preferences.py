"""
Used to preserve the state of the addon and its preferences.
e.g. location of include/ folder, conda installation for UV unwrapper, etc.

Referencing Blender's included Python API example:
https://github.com/blender/blender/blob/a35ce252c7f007daa8b9a4520f7aca8067313501/doc/python_api/examples/bpy.types.AddonPreferences.1.py
"""

import os
import pathlib
import sys

import bpy
import bpy.props
import bpy.types

from .common import ADDON_ID


def LocateUserCondaDirectory() -> str:
    """
    Used to find the location of the user's Conda directory via the user's system environment 
    variables.
    Should be executed only when the addon is first added.
    """
    # XXX: This may not be the safest operation
    environment_paths: list[str] = os.environ["PATH"].split(os.pathsep)

    # Search Conda directory in user's system environment paths
    for path in environment_paths:
        if path.lower().endswith("conda3") or path.lower().endswith("conda"):
            # Stop searching and return with the path we found.
            return path

    # If no paths found, then return error.
    raise OSError("Conda environment executable could not be located.\n"
                  "Please specify Conda directory in your system environment path variables.")


class RanatoPreferences(bpy.types.AddonPreferences):
    """
    Panel to appear in Blender's addon preferences page.
    """
    bl_label: str = "Ranato Preferences"
    bl_idname: str = ADDON_ID
    bl_space_type: str = 'PREFERENCES'
    bl_region_type: str = 'WINDOW'
    bl_context: str = "addons"

    _DIRECTORY_BASE_ADDON: pathlib.Path = pathlib.Path(__file__).parent
    _DIRECTORY_CONDA: str = LocateUserCondaDirectory()
    _FILEPATH_CAMPEN: pathlib.Path = _DIRECTORY_BASE_ADDON / \
        "bin" / "ConformalIdealDelaunay" / "script_conformal.py"
    _FILEPATH_CEPS: pathlib.Path = _DIRECTORY_BASE_ADDON / "bin" / "CEPS" / "parameterize.exe"
    _FILEPATH_MOVING_CONES: pathlib.Path = _DIRECTORY_BASE_ADDON / "bin" / "MovingCones" / "ConeGenes.exe"

    directory_temp: bpy.props.StringProperty(
        name="Script Temporary Directory",
        description="Temporary directory for script/executable I/O.",
        subtype='DIR_PATH',
        default=os.path.join(_DIRECTORY_BASE_ADDON, "__temp__")
    )

    directory_python: bpy.props.StringProperty(
        name="Blender Python Environment Directory",
        description="Directory for Blender's Python environment.",
        subtype='DIR_PATH',
        default=f"{sys.prefix}"
    )

    filepath_conda: bpy.props.StringProperty(
        name="Conda Executable Filepath",
        description="Filepath for Conda executable for UV unwrapper.",
        subtype='FILE_PATH',
        default=os.path.join(_DIRECTORY_CONDA, "envs", "cm_env", "python.exe")
    )

    # TODO: OK, really do make a UV unwrapper class because some of these (i.e. the CETM)
    #       actually does Python calls
    filepath_uv_unwrap_campen: bpy.props.StringProperty(
        name="Campen et al. 2021 UV Unwrapper Filepath",
        description="Directory for Campen et al 2021 UV unwrapper (aka mesh parametrization).",
        subtype='FILE_PATH',
        default=_FILEPATH_CAMPEN.as_posix()
    )

    filepath_uv_unwrap_ceps: bpy.props.StringProperty(
        name="CEPS UV Unwrapper Filepath",
        description="Directory for CEPS UV unwrapper (aka mesh parametrization).",
        subtype='FILE_PATH',
        default=_FILEPATH_CEPS.as_posix()
    )

    filepath_moving_cones: bpy.props.StringProperty(
        name="Moving Cones Filepath",
        description="Filepath for cone vertices executable.",
        subtype='FILE_PATH',
        default=_FILEPATH_MOVING_CONES.as_posix()
    )

    def draw(self, context: bpy.types.Context) -> None:
        """ Establishes Blender properties so that this data is held within Blender itself... roughly speaking.
        """
        layout: bpy.types.UILayout = self.layout
        layout.prop(self, "directory_python")
        layout.prop(self, "directory_temp")
        layout.prop(self, "filepath_conda")
        layout.prop(self, "filepath_uv_unwrap_campen")
        layout.prop(self, "filepath_uv_unwrap_ceps")
        layout.prop(self, "filepath_moving_cones")


class RANATO_OT_addon_preferences(bpy.types.Operator):
    """
    Display example preferences

    # See example:
    https://projects.blender.org/blender/blender/src/commit/1f60fcfed7decaee1859432d751326b830124812/doc/python_api/examples/bpy.types.AddonPreferences.1.py
    """
    bl_idname: str = "object.addon_preferences"
    bl_label: str = "Ranato add-on Preferences"
    bl_options: set = {'REGISTER', 'UNDO'}

    def execute(self, context: bpy.types.Context) -> set:
        """
        Executes the operator.
        Grabs from the context preferences and the addon's preferences.
        """
        addon_prefs: bpy.types.AddonPreferences | None = context.preferences.addons[ADDON_ID]

        if addon_prefs is None:
            print(f"Could not identify the addon preferences for {ADDON_ID}")
            return {'ERROR'}

        self.report({'INFO'}, "Successful addon retrieval")
        return {'FINISHED'}
