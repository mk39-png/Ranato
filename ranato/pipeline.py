"""
Keeps track of data throughout the addon. 

Implementation inspired from 
# https://github.com/Griperis/BlenderDataVis/blob/master/data_vis/data_manager.py 
"""

import subprocess
from typing import Any

import bpy.types
from mathutils import Matrix

# from .algebraic_contours.exec.generate_algebraic_contours import \
#     generate_algebraic_contours
from .common import DIRECTORY_TEMP, FILEPATH_UV_UNWRAPPER


def call_uv_unwrapper() -> None:
    """
    Calls the uv_unwrapper algorithm with specified commands.
    Also calls to activate the Conda environment if that has not been done already.
    """
    # Call sys subprocess to execute python script in another file.
    file_path_conda = bpy.context.preferences.addons[__package__].preferences.file_path_conda

    subprocess.run([file_path_conda,
                    FILEPATH_UV_UNWRAPPER,
                    "--input", DIRECTORY_TEMP,
                    "--fname", "temp.obj",
                    "--output", DIRECTORY_TEMP,
                    "--output_type", "param",
                    "--output_format", "obj"],
                   check=False)


def get_projection_matrix(context: bpy.types.Context) -> Matrix:
    """
    Retrieves the projection matrix for the current scene to use with Algebraic Contours generator.

    https://github.com/dfelinto/blender/blob/ec9977855f9264ecf6af5b4c8e6d10324a02028e/doc/python_api/
    examples/gpu.offscreen.1.py#L58-L64
    https://github.com/blender/blender/blob/main/doc/python_api/examples/gpu.9.py
    """

    # TODO: deal with case if camera does not exist within a scene

    depsgraph: bpy.types.Depsgraph = context.evaluated_depsgraph_get()
    scene: bpy.types.Scene | None = context.scene
    render: bpy.types.RenderSettings = scene.render
    camera: bpy.types.Object | None = scene.camera
    # print(camera.data.view_frame())

    modelview_matrix: Any | Matrix = camera.matrix_world.inverted()
    projection_matrix: Matrix = camera.calc_matrix_camera(
        depsgraph,
        x=render.resolution_x,
        y=render.resolution_y,
        scale_x=render.pixel_aspect_x,
        scale_y=render.pixel_aspect_y,
    )

    return projection_matrix


class OBJECT_OT_pipeline(bpy.types.Operator):
    """ 
    With the mesh saved to an .obj, we run the whole pipeline to calculate the 
    contours.
    """
    bl_idname: str = "object.pipeline"
    bl_label: str = "Contours Pipeline"
    # bl_property = "pipeline"

    def execute(self, context: bpy.types.Context) -> set:
        """
        Execute the operator.
        Runs the contour calculation pipeline.
        """
        # Now with these two in the temp folder, we can call the uv_unwrapper
        self.report({'INFO'}, "Calling UV unwrapper...")
        call_uv_unwrapper()

        # Now, with the UV unwrapped mesh, we can now call the main program for
        # processing the whole mesh... from another folder.
        projection_matrix: Matrix = get_projection_matrix(context)
        self.report({'INFO'}, "Generating algebraic contours...")
        # generate_algebraic_contours(projection_matrix)
        return {"FINISHED"}
