"""
UI layer.
"""

import math
import os
from typing import Any

import bpy
from bpy.props import EnumProperty
from bpy.types import Context, Mesh, Object

from .common import DIRECTORY_TEMP

# TODO: Implement future version utilize vertex position, texture coordinates, and face indices
#       of the mesh directly from Blender rather than an .obj file.


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


class OBJECT_OT_search_mesh_operator(bpy.types.Operator):
    """
    Brings up UI panel for searching for a particular mesh. 
    Then, returns the string key for the particular mesh. 
    """
    bl_idname = "object.search_mesh_operator"
    bl_label = "Search Mesh Operator"
    bl_property = "user_search"

    # https://blenderartists.org/t/menu-enumproperty/1446897
    user_search: EnumProperty(items=ops_get_objects)

    # https://docs.blender.org/api/current/bpy.types.Depsgraph.html

    def execute(self, context: Context) -> set:
        """
        Execute the operator.
        Grabs the objects within the dependency graph.
        """

        # TODO: only grab objects from the dependency graph (i.e. currently visible meshes)

        #
        # Retrieve the mesh object
        #
        # Based off user selection, retrieve reference to Blender object
        selected_object: Object = bpy.data.objects[self.user_search]
        selected_mesh: Mesh = bpy.data.meshes[self.user_search]

        #
        # Error handling when user selects a non-mesh
        #
        # https://surf-visualization.github.io/blender-course/api/meshes/#accessing-mesh-data-object-mode
        if selected_object.type == "MESH":
            # HACK: call operator to select mesh so that blender can utilize the
            # "export_selected_objects" flag to only export the desired mesh
            bpy.data.objects[selected_object.name].select_set(True)

            # TODO: check if there is already an .obj in temp.

            # TODO: the start and end frames should be dependent on the user selected start and
            #  end frames...
            # FIXME: file should be temporarily held in the addon's directory
            # https://github.com/benrugg/AI-Render/blob/main/analytics.py
            bpy.ops.wm.obj_export(filepath=os.path.join(DIRECTORY_TEMP, "temp.obj"),
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

            # Now, write the angle file for this mesh, defaulting at 2pi
            # print(len(selected_mesh.vertices))
            with open(os.path.join(DIRECTORY_TEMP, "temp_Th_hat"),
                      "w", encoding="utf8") as f:
                for _ in range(len(selected_mesh.vertices)):
                    f.write(f"{(math.pi * 2.0)}\n")

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
