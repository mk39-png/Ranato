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
from ..common import ADDON_ID


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


class OBJECT_OT_search_mesh_operator(bpy.types.Operator):
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

        #
        # Error handling when user selects a non-mesh
        #
        # https://surf-visualization.github.io/blender-course/api/meshes/#accessing-mesh-data-object-mode
        if selected_object.type == "MESH":
            # HACK: call operator to select mesh so that blender can utilize the
            # "export_selected_objects" flag to only export the desired mesh
            bpy.data.objects[selected_object.name].select_set(state=True)
            print("LOOK HERE", selected_object.data.vertices)

            # TODO: check if there is already an .obj in temp.

            # TODO: the start and end frames should be dependent on the user selected start and
            #  end frames...
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

            #
            # PREPARING FOR UV UNWRAPPING
            #
            # After running the executable for locating cone indices, be sure to save where they are.
            # Construct an array of size matching number of vertices
            vertex_angles: np.ndarray = np.full(shape=(len(selected_object.data.vertices)),
                                                fill_value=(np.pi * 2.0))

            # Now, get location of cone vertex angles (all the rows in the 0th column)
            cone_vertex_indices: np.ndarray = np.loadtxt(
                directory_temp / "temp-cones.txt", dtype=int)[:, 0]

            # NOTE: it seems that we may not need this?
            # At least testing with the bob duck mesh, using 2pi for the vertices worked just fine.
            # And it seems like a single island for the UV unwrapping is preferred to work fine.
            # Then, save the location of the cones into vertex_angles per Capouellez et al. 2023
            # vertex_angles[cone_vertex_indices] = np.pi    # * 3.0  # * random.random()

            # Finally, save to file...
            temp_file: pathlib.Path = pathlib.Path(directory_temp, "temp_Th_hat")
            np.savetxt(fname=temp_file, X=vertex_angles, newline="\n")

            # # TODO: move this functionality over to uv unwrap where it's more closely related.
            # # Now, write the angle file for this mesh, defaulting at 2pi
            # with open(temp_file, "w", encoding="utf8") as file:
            #     for _ in range(len(selected_object.data.vertices)):
            #         file.write(f"{(math.pi * 2.0)}\n")
            #         # file.write(f"{(math.pi * 1.0)}\n")

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
