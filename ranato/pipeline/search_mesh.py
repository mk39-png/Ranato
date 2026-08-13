"""
UI layer.
"""

import os
import pathlib
from typing import Any

import bpy
import bpy.props
import bpy.types
import numpy as np

# TODO: fix relative import
from ..common import ADDON_ID, INPUT_OBJ_FILENAME


# TODO: Implement future version utilize vertex position, texture coordinates, and face indices
#       of the mesh directly from Blender rather than an .obj file.
def ops_get_objects(self, context) -> list[Any]:
    """
    Grabs Blender ID of scene mesh.
    """
    enum: list[tuple[str, str, str]] = []

    # Getting data in the SCENE (rather than all objects in the file)
    for obj in bpy.context.scene.objects:
        id_ = str(obj.name)
        name: str = id_
        desc: str = "Description " + str(obj.name)

        if obj.type == "MESH":
            enum.append((id_, name, desc,))

    return enum


class RANATO_OT_Export_Mesh(bpy.types.Operator):
    """
    Exports selected mesh as .obj file to temporary directory
    """
    bl_idname = "object.export_mesh_operator"
    bl_label = "Export Mesh Operator"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        """ Checks if we can export a mesh (can only do so once mesh is selected).
        """
        return hasattr(context.scene, "target_mesh")

    # https://docs.blender.org/api/current/bpy.types.Depsgraph.html
    def execute(self, context: bpy.types.Context) -> set:
        """
        Execute the operator.
        Grabs the objects within the dependency graph.
        """

        # TODO: only grab objects from the dependency graph (i.e. currently visible meshes)
        directory_temp: str = pathlib.Path(
            bpy.context.preferences.addons[ADDON_ID].preferences.directory_temp)

        #
        # Retrieve the mesh object
        #
        # Based off user selection, retrieve reference to Blender object
        selected_object: bpy.types.Object = context.scene.target_mesh

        #
        # Error handling when user selects a non-mesh
        #
        # https://surf-visualization.github.io/blender-course/api/meshes/#accessing-mesh-data-object-mode
        # TODO: use an enum of type MESH rather than typing out "MESH"
        if selected_object.type == "MESH":
            # HACK: call operator to select mesh so that blender can utilize the
            # "export_selected_objects" flag to only export the desired mesh
            # bpy.data.objects[selected_object.name].select_set(state=True)
            bpy.context.scene.target_mesh.select_set(state=True)

            # TODO: check if there is already an .obj in temp.
            # FIXME: file should be temporarily held in the addon's directory
            # https://github.com/benrugg/AI-Render/blob/main/analytics.py
            bpy.ops.wm.obj_export(filepath=os.path.join(directory_temp, "temp.obj"),
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
                                  export_triangulated_mesh=True,
                                  export_curves_as_nurbs=False,
                                  export_object_groups=False,
                                  export_material_groups=False,
                                  export_vertex_groups=False,
                                  export_smooth_groups=False)
        else:
            # User mis-selected a non-mesh.
            self.report(
                {'ERROR'}, f"Attempted exporting \"{selected_object.name}\" of type {selected_object.type}, please select an object of type MESH.")
            return {"CANCELLED"}

        # All good, return success
        self.report({'INFO'}, f"Successfully exported MESH \"{selected_object.name}\"!")
        return {'FINISHED'}


class RANATO_OT_search_mesh_operator(bpy.types.Operator):
    """
    Brings up UI panel for searching for a particular mesh. 
    Then, returns the string key for the particular mesh. 
    """
    bl_idname = "object.search_mesh_operator"
    bl_label = "Search Mesh Operator"
    bl_property = "user_search"

    # https://blenderartists.org/t/menu-enumproperty/1446897
    user_search: bpy.props.EnumProperty(items=ops_get_objects)

    # https://docs.blender.org/api/current/bpy.types.Depsgraph.html
    def execute(self, context: bpy.types.Context) -> set:
        """
        Execute the operator.
        Grabs the objects within the dependency graph.
        """

        # TODO: only grab objects from the dependency graph (i.e. currently visible meshes)
        directory_temp: str = pathlib.Path(
            bpy.context.preferences.addons[ADDON_ID].preferences.directory_temp)

        #
        # Retrieve the mesh object
        #
        # Based off user selection, retrieve reference to Blender object
        selected_object: bpy.types.Object = bpy.context.scene.objects[self.user_search]
        # selected_object = bpy.context.scene.target_mesh
        # print("ME", selected_object)
        # print("ME", selected_object2)

        #
        # Error handling when user selects a non-mesh
        #
        # https://surf-visualization.github.io/blender-course/api/meshes/#accessing-mesh-data-object-mode
        if selected_object.type == "MESH":
            # HACK: call operator to select mesh so that blender can utilize the
            # "export_selected_objects" flag to only export the desired mesh
            bpy.data.objects[selected_object.name].select_set(state=True)
            print("LOOK HERE", selected_object.data.vertices)
            # bpy.context.scene.target_mesh.select_set(state=True)
            # TODO: check if there is already an .obj in temp.

            # TODO: the start and end frames should be dependent on the user selected start and
            #  end frames...
            # FIXME: file should be temporarily held in the addon's directory
            # https://github.com/benrugg/AI-Render/blob/main/analytics.py
            bpy.ops.wm.obj_export(filepath=os.path.join(directory_temp, INPUT_OBJ_FILENAME),
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
                                  export_triangulated_mesh=True,
                                  export_curves_as_nurbs=False,
                                  export_object_groups=False,
                                  export_material_groups=False,
                                  export_vertex_groups=False,
                                  export_smooth_groups=False)
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
        https://docs.blender.org/api/current/bpy.types.Operator.html#enum-search-popup
        """
        context.window_manager.invoke_search_popup(self)
        return {'RUNNING_MODAL'}
