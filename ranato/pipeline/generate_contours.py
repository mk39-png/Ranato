"""
Goes through the contour render pipeline of .obj mesh extraction, generating algebraic surface,
generating contours, generating contour .svg

Implementation inspired from
# https://github.com/Griperis/BlenderDataVis/blob/master/data_vis/data_manager.py
"""
import pathlib
import subprocess

import bpy.types
import numpy as np
from mathutils import Matrix
from pyalgcon.pipelines.generate_algebraic_contours import \
    generate_algebraic_contours


def blender_to_opengl_matrix(blender_camera_matrix: np.ndarray) -> np.ndarray:
    """
    Convert Blender to OpenGL-style coordinate system used by PYAC

    Referenced Visuals3D's post:
    https://github.com/facebookresearch/pytorch3d/issues/1105

    :param blender_matrix: 4x4 array where +Y is facing away, +Z is upwards, and +X is right
    :type c2w: np.ndarray

    # TODO: double check coordinate system converted
    :return: converted OpenGL matrix where +Y is up, +X is right, +Z is towards the camera 
    :rtype: np.ndarray
    """
    assert blender_camera_matrix.shape == (4, 4)
    opengl_camera_matrix = blender_camera_matrix

    # LOAD INTO POLYSCOPE AND SEE
    folder: pathlib.Path = pathlib.Path(__file__).parent
    np.savetxt(folder / "temp" / "temp_camera_matrix.csv",
               opengl_camera_matrix, delimiter=",", fmt="%f")

    # HACK: running venv python directly rather than using Blender's python env
    subprocess.run(
        [pathlib.Path(r"D:\Repos\Ranato\.venv\Scripts\python.exe"),  # sys.executable,
         str(folder / "run_polyscope.py"),
         "--file",
         str(folder / "temp" / "temp.obj"),
         "--camera",
         str(folder / "temp" / "temp_camera_matrix.csv")],
        check=True,
        # capture_output=True,
        # text=True
    )

    return opengl_camera_matrix


def get_matrices(context: bpy.types.Context) -> np.ndarray:
    """
    Retrieves the projection matrix for the current scene to use with Algebraic Contours generator.

    https://github.com/dfelinto/blender/blob/ec9977855f9264ecf6af5b4c8e6d10324a02028e/doc/python_api/examples/gpu.offscreen.1.py#L58-L64
    https://github.com/blender/blender/blob/main/doc/python_api/examples/gpu.9.py
    """

    # TODO: deal with case if camera does not exist within a scene
    depsgraph: bpy.types.Depsgraph = context.evaluated_depsgraph_get()
    scene: bpy.types.Scene | None = context.scene
    render: bpy.types.RenderSettings = scene.render
    camera: bpy.types.Object | None = scene.camera

    projection_matrix: Matrix = camera.calc_matrix_camera(
        depsgraph,
        x=render.resolution_x,
        y=render.resolution_y,
        scale_x=render.pixel_aspect_x,
        scale_y=render.pixel_aspect_y,
    )

    print("PROJECTION MAT: \n", projection_matrix)
    print("MAT WORLD: \n", camera.matrix_world)
    camera_matrix_pyac: np.ndarray = blender_to_opengl_matrix(
        np.array(camera.matrix_world))
    print("TRANSLATE TO ASOC: \n", camera_matrix_pyac)

    return camera_matrix_pyac


# Generate a PNG or whatnot and save to render layers...

# TODO: this is not the pipeline... it's morle like generation of contours
# The pipeline should really be the OVERALL steps of uv unwrapping, specifying cone vertices, etc
class OBJECT_OT_pipeline(bpy.types.Operator):
    """
    With the mesh saved to an .obj, we run the whole pipeline to calculate the
    contours.
    """
    bl_idname: str = "object.pipeline"
    bl_label: str = "Contours Pipeline"

    def execute(self, context: bpy.types.Context) -> set:
        """
        Execute the operator.
        Runs the contour calculation pipeline.
        """

        # Now, with the UV unwrapped mesh, we can now call the main program for
        # processing the whole mesh.
        camera_matrix = get_matrices(context)
        directory_temp: pathlib.Path = pathlib.Path(
            bpy.context.preferences.addons[__package__].preferences.directory_temp)
        generate_algebraic_contours(camera_matrix, directory_temp /
                                    "temp_out.obj")

        self.report({'INFO'}, message="Generated algebraic contours!")
        return {"FINISHED"}
