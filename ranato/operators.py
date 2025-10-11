import math
import os
import subprocess
import sys
from ctypes import py_object
from typing import Any

import bpy
from bpy.props import EnumProperty
from bpy.types import (Context, Depsgraph, Mesh, MeshUVLoopLayer, MeshVertices,
                       Object, RenderSettings, Scene)
from mathutils import Matrix, Vector

# Resources helping with understanding what EnumProperty is all about
# (and also understanding how "register" interacts with the rest of the addon)
# https://blender.stackexchange.com/questions/247695/invoke-search-popup-for-a-simple-panel
# https://docs.blender.org/api/current/bpy.types.Operator.html#enum-search-popup
# TODO: code that gets all meshes in the scene collection
# TODO: add operator that interfaces with contours code.
# TODO: grab camera matrix from scene.
# TODO: change this to only get objects within the dependency graph....


def call_uv_unwrapper() -> None:
    """
    Calls the uv_unwrapper algorithm with specified commands.
    Also calls to activate the Conda environment if that has not been done already.
    """


@staticmethod
def get_camera_matrix(context: Context) -> Matrix:
    """
    Retrieving camera matrix for the current scene to use with Algebraic Contours generator.

    https://github.com/dfelinto/blender/blob/ec9977855f9264ecf6af5b4c8e6d10324a02028e/doc/python_api/
    examples/gpu.offscreen.1.py#L58-L64
    https://github.com/blender/blender/blob/main/doc/python_api/examples/gpu.9.py
    """

    # TODO: deal with case if camera does not exist within a scene

    depsgraph: Depsgraph = context.evaluated_depsgraph_get()
    scene: Scene | None = context.scene
    render: RenderSettings = scene.render
    camera: Object | None = scene.camera
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


def ops_get_objects(self, context) -> list[Any]:
    """
    Grabs Blender ID of scene mesh.
    """
    enum: list[tuple[str, str, str]] = []

    for obj in bpy.data.collections["Collection"].all_objects:
        id_ = str(obj.name)
        name: str = id_
        desc: str = "Description " + str(obj.name)
        enum.append((id_, name, desc,))

    return enum


class SearchMeshOperator(bpy.types.Operator):
    """
    Brings up UI panel for searching for a particular mesh.
    TODO: move to UI-related file.
    """
    bl_idname = "object.search_mesh_operator"
    bl_label = "Search Mesh Operator"
    bl_property = "user_search"

    # https://blenderartists.org/t/menu-enumproperty/1446897
    user_search: EnumProperty(items=ops_get_objects)

    # TODO: Implement future version utilize vertex position, texture coordinates, and face indices
    #       of the mesh directly from Blender rather than an .obj file.

    # https://docs.blender.org/api/current/bpy.types.Depsgraph.html
    def execute(self, context: Context) -> set:
        """
        Execute the operator.
        Grabs the objects within the dependency graph.
        """
        #
        # Retrieve the camera.
        #
        projection_matrix: Matrix = get_camera_matrix(context)

        # TODO: save the projection matrix to a temporary file for the uv unwrapper... if needed.

        #
        # Retrieve the mesh object
        #
        # Based off user selection, retrieve reference to Blender object
        selected_object: Object = bpy.data.objects[self.user_search]
        selected_mesh: Mesh = bpy.data.meshes[self.user_search]

        # https://surf-visualization.github.io/blender-course/api/meshes/#accessing-mesh-data-object-mode
        if selected_object.type == "MESH":
            # HACK: call operator to select mesh so that blender can utilize the
            # "export_selected_objects" flag to only export the desired mesh
            bpy.data.objects[selected_object.name].select_set(True)

            # TODO: the start and end frames should be dependent on the user selected start and
            #  end frames...
            # FIXME: file should be temporarily held in the addon's directory
            # https://github.com/benrugg/AI-Render/blob/main/analytics.py
            base_directory: str = os.path.dirname(__file__)
            temp_directory: str = os.path.join(base_directory, "temp")
            bpy.ops.wm.obj_export(filepath=os.path.join(temp_directory, "temp_in.obj"),
                                  check_existing=True,
                                  start_frame=0,
                                  end_frame=0,
                                  export_selected_objects=True,
                                  forward_axis='NEGATIVE_Z',  # TODO: may need to change these
                                  up_axis='Y',  # TODO: may need to change these
                                  export_colors=False,
                                  export_uv=True,
                                  export_normals=False,
                                  export_materials=False,
                                  export_triangulated_mesh=False,
                                  export_curves_as_nurbs=False,
                                  export_object_groups=False,
                                  export_material_groups=False,
                                  export_vertex_groups=False,
                                  export_smooth_groups=False)

            # TODO: check if mesh has already been UV unwrapped or not... or check if the button for
            # UV unwrapping has been enabled or not.
            # python_exe: str = os.path.join(sys.prefix, 'bin', 'python.exe')
            # print(python_exe)

            # Now, write the angle file for this mesh, defaulting at 2pi
            print(len(selected_mesh.vertices))
            with open(os.path.join(temp_directory, "temp_Th_hat"),
                      "w", encoding="utf8") as f:
                for _ in range(len(selected_mesh.vertices)):
                    f.write(f"{math.pi * 2.0}\n")

            # Now with these two in the temp folder, we can call the uv_unwrapper
            # Call sys subprocess to execute python script in another file.
            uv_unwrapper_directory: str = os.path.join(
                base_directory, "uv_unwrapper", "script_conformal.py")
            # FIXME: make this non-hard coded
            uv_unwrapper_env_directory: str = (
                "C:\\Users\\newbe\\miniconda3\\envs\\cm_env_original\\python.exe")
            subprocess.run([uv_unwrapper_env_directory,
                            os.path.join(base_directory, "uv_unwrapper", "script_conformal.py"),
                            "--input", temp_directory,
                            "--fname", "temp_in.obj",
                            "--output", temp_directory,
                            "--output_type", "param",
                            "--output_format", "obj"])

            # Now, with the UV unwrapped mesh, we can now call the main program for
            # processing the whole mesh... from another folder.

        else:
            # User mis-selected a non-mesh.
            self.report({'ERROR'}, "Improper selection (select a mesh): " + self.user_search)
            return {"CANCELLED"}

        # All good, return success
        self.report({'INFO'}, "Selected: " + self.user_search)
        return {'FINISHED'}

    def invoke(self, context, event) -> set:
        """
        Invokes the operator.
        """
        context.window_manager.invoke_search_popup(self)
        return {'RUNNING_MODAL'}


# Logistics functions
def register() -> None:
    """
    Register SearchEnumOperator class.
    """
    bpy.utils.register_class(SearchMeshOperator)


def unregister() -> None:
    """
    Unregister SearchEnumOperator class.
    """
    bpy.utils.unregister_class(SearchMeshOperator)


if __name__ == "__main__":
    register()
