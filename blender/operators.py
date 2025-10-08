from typing import Any

import bpy
import numpy as np
from bpy.props import EnumProperty
from bpy.types import (Depsgraph, Mesh, MeshUVLoopLayer, MeshVertices, Object,
                       RenderSettings, Scene)
from mathutils import Matrix, Vector

# Resources helping with understanding what EnumProperty is all about
# (and also understanding how "register" interacts with the rest of the addon)
# https://blender.stackexchange.com/questions/247695/invoke-search-popup-for-a-simple-panel
# https://docs.blender.org/api/current/bpy.types.Operator.html#enum-search-popup
# TODO: code that gets all meshes in the scene collection
# TODO: add operator that interfaces with contours code.
# TODO: grab camera matrix from scene.
# TODO: change this to only get objects within the dependency graph....


# https://github.com/benrugg/AI-Render/blob/main/analytics.py
# base = os.path.dirname(__file__)
# module_dir: str = os.path.join(base, "dependencies")
# sys.path.append(module_dir)


# def read_mesh_obj() -> None:
#     """
#     Reads a .obj file of the selected mesh in the current project directory.
#     """


# def write_mesh_obj() -> None:
#     """
#     Saves a .obj file of the selected mesh to current project directory.
#     """
#     # https://blender.stackexchange.com/questions/84934/what-is-the-python-script-to-export-the-selected-meshes-in-obj/309888#309888
# bpy.ops.wm.obj_export(filepath="temp.obj",
#                       check_existing=True,
#                       start_frame=0,
#                       end_frame=0,
#                       export_uv=True,
#                       export_normals=True,
#                       export_selected_objects=True,
#                       forward_axis='NEGATIVE_Z',  # TODO: may need to change these
#                       up_axis='Y',  # TODO: may need to change these
#                       export_triangulated_mesh=False,
#                       export_curves_as_nurbs=False,
#                       export_object_groups=False,
#                       export_material_groups=False,
#                       export_vertex_groups=False,
#                       export_smooth_groups=False)


# @staticmethod
# def get_camera_matrix() -> Matrix:
#     """
#     Retrieving camera matrix for the current scene to use with Algebraic Contours generator.

#     https://github.com/dfelinto/blender/blob/ec9977855f9264ecf6af5b4c8e6d10324a02028e/doc/python_api/
#     examples/gpu.offscreen.1.py#L58-L64
#     https://github.com/blender/blender/blob/main/doc/python_api/examples/gpu.9.py
#     """

#     # TODO: deal with case if camera does not exist within a scene

#     context: Context = bpy.context
#     scene: Scene | None = context.scene
#     render: RenderSettings = scene.render
#     camera: Object | None = scene.camera

#     modelview_matrix: Any | Matrix = camera.matrix_world.inverted()
#     projection_matrix: Matrix = camera.calc_matrix_camera(
#         render.resolution_x,
#         render.resolution_y,
#         render.pixel_aspect_x,
#         render.pixel_aspect_y,
#     )

#     return projection_matrix


def get_objects(self, context) -> list[Any]:
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
    bl_property = "my_search"

    # https://blenderartists.org/t/menu-enumproperty/1446897
    my_search: EnumProperty(items=get_objects)

    # TODO: get the active camera as well...
    # TODO: get the UV coordinates?
    # TODO: get the vertex coordinates...

    # TODO: Implement future version utilize vertex position, texture coordinates, and face indices
    #       of the mesh directly from Blender rather than an .obj file.

    # HACK:

    # https://docs.blender.org/api/current/bpy.types.Depsgraph.html
    def execute(self, context) -> set:
        """
        Execute the operator.
        Grabs the objects within the dependency graph.
        """
        depsgraph: Depsgraph = context.evaluated_depsgraph_get()

        # NOTE: all of the values below should match the .obj file... for the most part.
        # TODO: this might fail if user is in edit mode.
        # me: Mesh = bpy.context.object.data

        # uv_layer = me.uv_layers.active.uv  # uv coordinates
        # print(me.vertices)  # float values for vertex positions.
        # for poly in me.polygons:
        #     poly.index  # face index
        #     poly.vertices  # vertex index
        # print(context.object)

        print()

        print(context.scene)

        scene: Scene | None = context.scene
        render: RenderSettings = scene.render
        camera: Object | None = scene.camera

        print("CAMERA INFO:")
        print(camera.data.view_frame())
        modelview_matrix: Any | Matrix = camera.matrix_world.inverted()
        print(modelview_matrix)
        projection_matrix: Matrix = camera.calc_matrix_camera(
            depsgraph,
            x=render.resolution_x,
            y=render.resolution_y,
            scale_x=render.pixel_aspect_x,
            scale_y=render.pixel_aspect_y,
        )
        print(projection_matrix)
        print("DIVIDER")

        for object_instance in depsgraph.object_instances:
            # This is an object which is being instanced.
            obj: Object | None = object_instance.object

            print("Look here")
            # TODO: add check to see if "MESH" has "uv" coordinates.

            # https://surf-visualization.github.io/blender-course/api/meshes/#accessing-mesh-data-object-mode
            if obj.type == "MESH":
                selected_mesh: Mesh = bpy.data.meshes[obj.name]
                bpy.data.objects[obj.name].select_set(True)
                # uv_coordinates = selected_mesh.uv_layers.active.uv
                uv_map: MeshUVLoopLayer | None = selected_mesh.uv_layers.active

                # HACK: call operator to select mesh so that blender can utilize the
                # "export_selected_objects" flag to only export the desired mesh
                # TODO: Call the operato
                # bpy.ops.
                # bpy.ops.object.select

                bpy.ops.wm.obj_export(filepath="temp.obj",
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

                # OK, we got MOST of the things we need.
                # uv_coordinates: list[Vector] = [v.uv for v in uv_map.data]
                # vertex_indices: list[int] = [v.vertices for v in selected_mesh.polygons]
                # face_indices: list[int] = [v.index for v in selected_mesh.polygons]
                # vertex_coordinates: list[Vector] = [v.co for v in selected_mesh.vertices]
                # print(np.unique(np.array(uv_coordinates), axis=0).shape)

                # print(face_indices)
                # print(len(vertex_coordinates))
                # for i in range(5):
                #     print(vertex_coordinates[i][0])
                #     print(vertex_coordinates[i][1])
                #     print(vertex_coordinates[i][2])
                #     print()

                # FIXME: potentially bad uv_coordinates extraction. as in, not correct with obj file
                # That's because the UV_coordinates are duplicates for vertices.
                # As in, it's 3x more than we want.
                # # What if use loop index?
                # uv_coord_test = []
                # for tri in selected_mesh.loop_triangles:
                #     for loop_index in tri.loops:
                #         # print(uv_map.data[loop_index].uv)
                #         uv_coord_test.append(uv_map.data[loop_index].uv)
                # print(len(uv_coord_test))

                # print(len(uv_coordinates))
                # for i in range(5):
                #     print(uv_coordinates[i][0])
                #     print(uv_coordinates[i][1])
                #     print()

                # for i in range(5):
                #     print(vertex_indices[i][0])
                #     print(vertex_indices[i][1])
                #     print(vertex_indices[i][2])
                #     print()

            # print(obj.data.vertices[0])

            # `is_instance` denotes whether the object is coming from instances (as an opposite of
            # being an emitting object. )
            if not object_instance.is_instance:
                print(f"Object {obj.name} at {object_instance.matrix_world}")
            else:
                # Instanced will additionally have fields like uv, random_id and others which are
                # specific for instances. See Python API for DepsgraphObjectInstance for details,
                print(
                    f"Instance of {obj.name} at {object_instance.matrix_world}")

        self.report({'INFO'}, "Selected:" + self.my_search)
        return {'FINISHED', self.my_search}

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
