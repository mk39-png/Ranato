"""
Used to preserve the state of the addon and its preferences.
e.g. location of include/ folder, conda installation for UV unwrapper, etc.

Referencing Blender's included Python API example:
https://github.com/blender/blender/blob/a35ce252c7f007daa8b9a4520f7aca8067313501/doc/python_api/examples/bpy.types.AddonPreferences.1.py
"""

import os
import sys

import bpy
import bpy.props
import bpy.types


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
    bl_idname: str = __package__
    bl_space_type: str = 'PREFERENCES'
    bl_region_type: str = 'WINDOW'
    bl_context: str = "addons"

    # directory_path_conda: str = LocateUserCondaDirectory()

    # TODO: find purpose of having a place to specify Blender's Python directory.
    directory_path_python: bpy.props.StringProperty(
        name="Blender Python Environment Directory",
        description="Directory for Blender's Python environment.",
        subtype='DIR_PATH',
        default=f"{sys.prefix}"
    )

    # TODO: change to support only cm_env if the user followed the original UV unwrapper paper's setup
    # file_path_conda: bpy.props.StringProperty(
    #     name="Conda Executable Filepath",
    #     description="Filepath for Conda executable for UV unwrapper.",
    #     subtype='FILE_PATH',
    #     default=os.path.join(directory_path_conda, "envs", "cm_env_original", "python.exe")
    # )

    # TODO: allow user to specify "include/" folder to copy over and into Blender's Python Env
    #       to install Cholespy (because Blender's default Python ENV does not have include/ for some
    # reason)
    # XXX: this is for the legacy version of adding dependencies.
    # Newer versions of Ranato should support wheels.
    # FIXME: have place for wheels for Blender's new way of handling dependencies.
    directory_path_include: bpy.props.StringProperty(
        name="Python Environment Include Directory",
        description="include/ directory needed for installation of Cholespy via pip.",
        subtype='DIR_PATH',
        default=os.path.join(sys.prefix, "include")
    )

    def draw(self, context: bpy.types.Context) -> None:
        layout: bpy.types.UILayout = self.layout
        layout.prop(self, "directory_path_python")
        layout.prop(self, "file_path_conda")
        layout.prop(self, "directory_path_include")


class OBJECT_OT_addon_preferences(bpy.types.Operator):
    """
    Display example preferences
    """
    bl_idname: str = "object.addon_preferences"
    bl_label: str = "Ranato add-on Preferences"
    bl_options: set = {'REGISTER', 'UNDO'}

    def execute(self, context: bpy.types.Context) -> set:
        """
        Executes the operator.
        Grabs from the context preferences and the addon's preferences.
        """
        addon_prefs: bpy.types.AddonPreferences | None = context.preferences.addons[__package__]

        if addon_prefs is None:
            print(f"Could not identify the addon preferences for {__package__}")
            return {'ERROR'}

        self.report({'INFO'}, "Successful addon retrieval")
        return {'FINISHED'}
