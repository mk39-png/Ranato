"""
UI layer.
"""

import subprocess
from abc import ABC

import bpy
import bpy.props
import bpy.types


# TODO: separate blender-facing files and the regular files...
def locate_cones() -> None:
    """ Finds cone vertices on the mesh for use in UV unwrapping.

    Returns:
        _type_: _description_
    """

# TODO: Implement future version utilize vertex position, texture coordinates, and face indices
#       of the mesh directly from Blender rather than an .obj file.

# TODO: make UV unwapper class since we have 3 UV unwrappers with their own parameters...
# Like vertex angles and whatnot...


def call_uv_unwrapper() -> None:
    """
    # TODO: refactor so that function is universal for any UV unwrapper
    # TODO: this means passing in the args and params and whatnot...

    Calls the uv_unwrapper algorithm with specified commands.
    Also calls to activate the Conda environment if that has not been done already.
    """
    # Call sys subprocess to execute python script in another file.
    filepath_conda: str = bpy.context.preferences.addons[__package__].preferences.filepath_conda
    file_path_uv_unwrapper: str = bpy.context.preferences.addons[__package__].preferences.filepath_uv_unwrap_campen
    directory_temp: str = bpy.context.preferences.addons[__package__].preferences.directory_temp

    # TODO: have a check to make sure that file_path_uv_unwrapper preference has been checked.
    print(directory_temp)
    # TODO: define UV unwrapper filepath in one of the preferences addon menu boxes...

    # TODO: have a way to save the parameters/settings, especially with a mesh and its user-inputted cone vertices.
    # TODO: also for CEPS, need a way to track its parameters as well

    # TODO: allow user to specify parameters into this... by writing down whatever in the panel box
    # TODO: have fields in Ranato that allows user to specify these args...
    subprocess.run([filepath_conda,
                    file_path_uv_unwrapper,
                    "--input", directory_temp,
                    "--fname", "temp.obj",
                    # "--max_itr", "500",
                    "--output", directory_temp,
                    # "--no_round_Th_hat",
                    "--error_log",
                    "--output_type", "param",
                    "--output_format", "obj"
                    ],
                   check=False)


class MeshParametrizer(ABC):
    """ 
    Abstract class providing structure for any mesh parametrization (UV unwrapping) algorithm.
    Strategy pattern.
    """
    id: str = ""
    label: str = ""
    directory_temp: str = ""

    #
    # VARIABLES
    # Python installation location
    # Executable if applicable
    # Conda location if applicable
    # Foldernames
    # Name of the parametrization
    # Output directory
    # Input directory

    # METHODS:
    # Call UV unwrapper
    # UV unwrapper arguments (**args)
    # Can be flexible with a dict or something or passing in args
    # UV unwrapper parameters (cone vertices, distortion angles, etc)
    # Some interface for storing values between Blender and this
    # Some exporter of values stored (as a numpy CSV or whatnot...)
    #

    # Needs the script and its parameters to call in **args
    # Also, file to specify cone vertices if applicable


class OBJECT_OT_uv_unwrap(bpy.types.Operator):
    """
    Brings up UI panel for searching for a particular mesh. 
    Then, returns the string key for the particular mesh. 
    """
    bl_idname = "object.uv_unwrap"
    bl_label = "UV Unwrap"

    # https://docs.blender.org/api/current/bpy.types.Depsgraph.html
    def execute(self, context: bpy.types.Context) -> set:
        """
        Execute the operator.
        Grabs the objects within the dependency graph.
        """

        self.report({'INFO'}, "Calling UV unwrapper...")
        call_uv_unwrapper()

        # All good, return success
        self.report({'INFO'}, message="Generated UV unwrapping!")
        return {'FINISHED'}


class RANATO_UL_ItemList(bpy.types.UIList):
    """ UIList subclass. But is more concerned with the data stored in each property rather than 
    how it is displayed.
    """

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index) -> None:
        """ Draws each item in the list.

        More context about this function in the following:
        https://docs.blender.org/api/current/bpy.types.UIList.html

        Args:
            context (_type_): _description_
            layout (_type_): _description_
            data (_type_): RNA object containing the collection
            item (_type_): current drawn item of the collection
            icon (_type_): "computed" icon for the item (as an integer since some objects like materials or textures have custom icons ID that are not available as enum items)
            active_data (_type_): RNA object containing the active property for the collection (i.e. integer active item of the collection)
            active_propname (_type_): name of the active property 
            index (_type_): index of the current item in the collection
        """
        custom_icon = "OBJECT_DATAMODE"

        if self.layout_type in {"DEFAULT", "COMPACT"}:
            layout.label(text=item.name, icon=custom_icon)
            layout.prop(item, "vertex_index", text="Vertex Index")
            layout.prop(item, "angle", text="Vertex Angle")
        elif self.layout_type in {"GRID"}:
            layout.alignment = "CENTER"
            layout.label(text=f"{item.vertex_index}", icon=custom_icon)


# Setting up addon with multiple files
# https://blender.stackexchange.com/questions/202570/multi-files-to-addon

# Basic panel setup
# https://docs.blender.org/api/current/bpy.types.Panel.html

# Referencing below to see how Render Engine interacts with Panel
# (and changing it all so that the panel only appears when user activates Ranato)
# https://github.com/bnpr/Malt/blob/725f509ab25be736cb592cf1e9d5258ed4271e8a/BlenderMalt/MaltMaterial.py#L98
# https://github.com/Griperis/BlenderDataVis/blob/master/data_vis/operators/surface_chart.py


class VertexAngleItem(bpy.types.PropertyGroup):
    """ 
    Custom data class to hold vertices and their assigned angles (or )

    Useful sources below:
    https://sinestesia.co/blog/tutorials/using-uilists-in-blender/
    https://docs.blender.org/api/current/bpy.types.UIList.html
    https://blender.stackexchange.com/questions/15917/populate-a-list-with-custom-property-dictionary-data
    """

    # Properties to hold onto the data first.
    vertex_index: bpy.props.IntProperty(
        name='vertex_index',
        description="Selected vertex index",
        default=0,
        min=0
    )

    angle: bpy.props.FloatProperty(
        name="vertex_angle",
        description="Specified angle constraint for selected vertex",
        default=0.0,
        subtype="ANGLE"
    )
