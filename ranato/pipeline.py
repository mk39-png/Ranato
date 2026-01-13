"""
Goes through the contour render pipeline of .obj mesh extraction, generating algebraic surface,
generating contours, generating contour .svg

Implementation inspired from
# https://github.com/Griperis/BlenderDataVis/blob/master/data_vis/data_manager.py
"""
import multiprocessing
import pathlib
import subprocess
import polyscope

import bpy.types
import numpy as np
from mathutils import Matrix
from pyalgcon.contour_network.contour_network import ContourNetwork
from pyalgcon.exec.generate_algebraic_contours import \
    generate_algebraic_contours

from .common import DIRECTORY_TEMP

# def call_uv_unwrapper() -> None:
#     """
#     Calls the uv_unwrapper algorithm with specified commands.
#     Also calls to activate the Conda environment if that has not been done already.
#     """
#     # Call sys subprocess to execute python script in another file.
#     file_path_conda = bpy.context.preferences.addons[__package__].preferences.file_path_conda

#     subprocess.run([file_path_conda,
#                     FILEPATH_UV_UNWRAPPER,
#                     "--input", DIRECTORY_TEMP,
#                     "--fname", "temp.obj",
#                     "--output", DIRECTORY_TEMP,
#                     "--output_type", "param",
#                     "--output_format", "obj"],
#                    check=False)


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

    # Swaps the the Y and Z rows (for original ASOC program)
    # swap_y_z = np.identity(4)

    # MATRIX 0 TODO: might need something like -1 on the x axis as well.
    # swap_y_z = np.array([
    #     [1, 0, 0, 0],
    #     [0, -1, 0, 0],
    #     [0, 0, -1, 0],
    #     [0, 0, 0, 1]], dtype=np.float64)

    # MATRIX 1
    swap_y_z = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]], dtype=np.float64)

    # MATRIX 2
    # With just this, the translation to openGL seems fine
    # swap_y_z = np.array([
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, -1, 0],
    #     [0, 0, 0, 1]], dtype=np.float64)

    # NOTE: blender to opengl translation is already as fine as it is
    # swap_y_z: np.ndarray = np.array([
    #     [1, 0, 0, 0],
    #     [0, 0, -1, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 0, 1]])

    theta: float = np.deg2rad(180)
    blender_camera_matrix_swapped_y_z = swap_y_z @ blender_camera_matrix

    # This camera matrix is proper and works for PYAC
    # return np.array(((1.0000,  0.0000,  0.0000, 0.0000),
    #                  (0.0000, -0.7071,  -0.7071, 0.0000),
    #                  (0.0000,  -0.7071,  0.7071, 10.000),
    #                  (0.0000,  0.0000,  0.0000, 1.0000),))
    theta2 = np.deg2rad(180)
    rot_y = np.array([
        [np.cos(theta2), 0, np.sin(theta2)],
        [0, 1, 0],
        [-np.sin(theta2), 0, np.cos(theta2)]
    ])

    rot_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta2), -np.sin(theta2)],
        [0, np.sin(theta2), np.cos(theta2)]
    ])

    # Flip image rightside up (otherwise image will be upside down)
    # TODO: experiment with changing ccw to cw
    rot_z: np.ndarray = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]])

    # TODO: check CW and CCW rotation of that's being done
    # Performs rotation in world space rather than camera space.
    translation_vector: np.ndarray = blender_camera_matrix_swapped_y_z[:3, -1]  # Vector 3 elements
    rotation_frame_matrix: np.ndarray = np.linalg.inv(blender_camera_matrix_swapped_y_z[:3, :3])
    # rotation_frame_matrix = rotation_frame_matrix @ rot_y
    rotation_frame_matrix = rotation_frame_matrix @ rot_z
    # rotation_frame_matrix = rotation_frame_matrix @ rot_x
    translation_vector = translation_vector @ rotation_frame_matrix  # Make translation local

    # Assemble the extrinsic matrix (with the frame and the translation)
    opengl_camera_matrix: np.ndarray = np.zeros(shape=(4, 4), dtype=np.float64)
    opengl_camera_matrix[:3, :3] = np.linalg.inv(rotation_frame_matrix)  # .T
    opengl_camera_matrix[:3, -1] = translation_vector
    opengl_camera_matrix[3, 3] = 1  # set bottom right corner to 1 because homogeneous matrix

    # LOAD INTO POLYSCOPE AND SEE
    folder: pathlib.Path = pathlib.Path(__file__).parent
    np.savetxt(folder / "temp" / "temp_camera_matrix.csv",
               opengl_camera_matrix, delimiter=",", fmt="%f")
    # HACK: running venv python directly rather than using Blender's python env
    subprocess.run(
        [pathlib.Path("D:\Repos\Ranato\.venv\Scripts\python.exe"),  # sys.executable,
         str(folder / "run_polyscope.py"),
         "--file",
         str(folder / "temp" / "temp_out.obj"),
         "--camera",
         str(folder / "temp" / "temp_camera_matrix.csv")],
        check=True,
        # capture_output=True,
        # text=True
    )

    return opengl_camera_matrix  # np.linalg.inv(opengl_camera_matrix)


def get_matrices(context: bpy.types.Context) -> tuple[np.ndarray, np.ndarray]:
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

    projection_matrix: Matrix = camera.calc_matrix_camera(
        depsgraph,
        x=render.resolution_x,
        y=render.resolution_y,
        scale_x=render.pixel_aspect_x,
        scale_y=render.pixel_aspect_y,
    )

    # TODO: utilize the matrix parent inverse?
    # camera.matrix_parent_inverse
    # projection_matrix.invert()

    print("PROJECTION MAT: \n", projection_matrix)
    print("MAT WORLD: \n", camera.matrix_world)
    resulting: np.ndarray = blender_to_opengl_matrix(np.array(camera.matrix_world))
    print("TRANSLATE TO ASOC: \n", resulting)

    # TODO: remove the return of the projection_matrix since ASOC uses a special projection transformation
    # (i.e. project_inf_to_origin() or whatever that function was called)
    return np.array(projection_matrix), resulting


# Generate a PNG or whatnot and save to render layers...

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
        #
        # DEPRECATED: rely solely on the fact that the mesh has been UV unwrapped already
        #
        # Now with these two in the temp folder, we can call the uv_unwrapper
        # self.report({'INFO'}, "Calling UV unwrapper...")
        # call_uv_unwrapper()

        # Now, with the UV unwrapped mesh, we can now call the main program for
        # processing the whole mesh.
        projection_matrix, camera_matrix = get_matrices(context)
        generate_algebraic_contours(camera_matrix, DIRECTORY_TEMP /
                                    "temp_out.obj", projection_matrix)

        self.report({'INFO'}, "Generated algebraic contours!")
        return {"FINISHED"}
