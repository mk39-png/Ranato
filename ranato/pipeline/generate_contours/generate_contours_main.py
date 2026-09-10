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
from pyalgcon.contour_network.compute_intersections import \
    IntersectionParameters
from pyalgcon.contour_network.contour_network import (InvisibilityMethod,
                                                      InvisibilityParameters)
from pyalgcon.pipelines.generate_algebraic_contours import \
    generate_algebraic_contours
from pyalgcon.quadratic_spline_surface.optimize_spline_surface import \
    OptimizationParameters
from pyalgcon.utils.projected_curve_networks_utils import SVGOutputMode

from ...common import ADDON_ID, DEBUG


def opengl_to_pyac_matrix(opengl_camera_matrix_ref: np.ndarray) -> np.ndarray:
    """ 
    Converts OpenGL-style coordinate system to that used by PYAC.
    """
    # FIXME: potentially incorrect mirroring
    opengl_camera_matrix_ref[0] *= -1
    opengl_camera_matrix_ref[2] *= -1
    return np.copy(opengl_camera_matrix_ref)


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

    deg180 = np.deg2rad(180)
    rot_z = np.array([
        [np.cos(deg180), -np.sin(deg180), 0],
        [np.sin(deg180), np.cos(deg180), 0],
        [0, 0, 1]])

    c2w = np.copy(opengl_camera_matrix)
    swap_y_z = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]])

    c2w = swap_y_z @ c2w
    t = c2w[:3, -1]  # Extract translation of the camera
    r = c2w[:3, :3]  @ rot_z  # Extract rotation matrix of the camera
    t = t @ r  # Make rotation local

    opengl_camera_matrix = np.identity(4)
    opengl_camera_matrix[:3, :3] = r.T
    opengl_camera_matrix[:3, 3] = t
    opengl_camera_matrix[2] *= -1

    # NOTE: rounding to avoid close-to-zero floats
    # NOTE: adding 0.0 to avoid negative 0s
    opengl_camera_matrix = opengl_camera_matrix.round(5) + 0.0

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

    print("Blender PROJECTION MATRIX: \n", projection_matrix)
    print("Blender WORLD MATRIX: \n", camera.matrix_world)

    # TODO: move camera conversion somewhere else?
    # get_matrices() should only have the sole purpose of retrieving the current camera.
    opengl_camera_matrix: np.ndarray = blender_to_opengl_matrix(
        np.array(camera.matrix_world))
    pyac_camera_matrix: np.ndarray = opengl_to_pyac_matrix(opengl_camera_matrix)

    if DEBUG:
        directory_temp: pathlib.Path = pathlib.Path(
            bpy.context.preferences.addons[ADDON_ID].preferences.directory_temp)
        np.savetxt(directory_temp / "temp_camera_matrix.csv",
                   opengl_camera_matrix, delimiter=",", fmt="%f")

    if DEBUG:
        # HACK: running venv python directly rather than using Blender's python env
        subprocess.run(
            [pathlib.Path(r"D:\Repos\Ranato\.venv\Scripts\python.exe"),  # sys.executable,
             (pathlib.Path(__file__).parent.parent / "run_polyscope.py").as_posix(),
             "--file",
             (pathlib.Path(__file__).parent.parent / "__temp__" / "temp_out.obj").as_posix(),
             "--camera",
             (pathlib.Path(__file__).parent.parent / "__temp__" / "temp_camera_matrix.csv")],
            check=True,
            # capture_output=True,
            # text=True
        )

    return pyac_camera_matrix


class RANATO_OT_pipeline(bpy.types.Operator):
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

        # TODO: to generate contours, the previous steps of the pipeline MUST be complete.
        # As in...
        # * Export OBJ mesh
        # * Gathered UV unwrapping
        # * Set parameters for UV unwrapping
        # * Set parameters for algebraic contours
        # After all of that, we are able to proceed with generating algebraic contours.

        camera_matrix: np.ndarray = get_matrices(context)
        directory_temp: pathlib.Path = pathlib.Path(
            bpy.context.preferences.addons[ADDON_ID].preferences.directory_temp)

        # TODO: define arguments...
        # TODO: somehow set assertions to false.
        # NOTE: if generate_algebraic_contours is taking a LONG time for small meshes, it is likely that the camera is wrong.
        generate_algebraic_contours(camera_matrix, directory_temp /
                                    "temp_out.obj")

        # TODO: Generate a PNG or whatnot and save to render layers...
        self.report({'INFO'}, message="Successfully generated algebraic contours!")
        return {"FINISHED"}
