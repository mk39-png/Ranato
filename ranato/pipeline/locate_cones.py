import pathlib
import subprocess

import bpy
import bpy.types
from bpy.props import FloatProperty

from ..common import ADDON_ID


class RANATO_OT_locate_cones(bpy.types.Operator):
    """"""
    bl_idname = "ranato.locate_cones"
    bl_label = ""
    bl_description = ""
    bl_options = {"REGISTER", "UNDO"}

    # TODO: need a property associated with this operator...
    # Which is the distortion scale factor or soemthing like that

    def execute(self, context) -> set[str]:
        # NOTE: use context argument to avoid calling bpy.context globally
        preferences: bpy.types.AddonPreferences = context.preferences.addons[ADDON_ID].preferences
        directory_temp: pathlib.Path = pathlib.Path(preferences.directory_temp)
        filepath_moving_cones: pathlib.Path = pathlib.Path(preferences.filepath_moving_cones)
        distortion: float = context.scene.locate_cones_distortion

        subprocess.run([
            filepath_moving_cones.as_posix(),
            (directory_temp / "temp.obj").as_posix(),  # input mesh
            (directory_temp / "temp").as_posix(),  # cone filename
            f"{distortion}"],
            check=True
        )

        return {"FINISHED"}


# This has a certain arugment format to it...
# Something like the below...
# The ./out is like ./out-cones.txt or something like that
# It's automatically named
# But this is cool since it gives us cone vertices we can mess with
# ConeGenes.exe ./spot_control_mesh.obj ./out 0.2
