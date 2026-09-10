import pathlib

import bpy.props
import bpy.types
import numpy as np
from bpy_extras.io_utils import ExportHelper, ImportHelper

from ..common import ADDON_ID


def retrieve_cone_vertex_angles(vertex_angles: bpy.types.bpy_prop_collection_idprop,
                                ceps_format: bool = False):
    """ Retrieves ALL indices and vertices from vertex angles object

    Args:
        vertex_angles (bpy.types.bpy_prop_collection): _description_

    Returns:
        _type_: _description_
    """
    indices = np.fromiter((vertex.index for vertex in vertex_angles), dtype=int)
    angles = np.fromiter((vertex.angle for vertex in vertex_angles), dtype=float)
    return (indices, angles)


def retrieve_vertex_angles(context: bpy.types.Context):
    """ Retrieves vertex angles for each index of selected mesh based on
    default vertex angle and specified cone vertices.

    Args:
        context (Context): _description_
    """
    # NOTE: need selected mesh so that we know how many vertex angles to make (i.e. need the number of vertices of the mesh)
    if context.scene.target_mesh is None:
        raise ValueError("No mesh has been selected! Please select a mesh to process")

    #
    # PREPARING FOR UV UNWRAPPING
    #
    selected_object: bpy.types.Object = context.scene.target_mesh

    # After running the executable for locating cone indices, be sure to save where they are.
    # Construct an array of size matching number of vertices
    vertex_angles: np.ndarray = np.full(shape=(len(selected_object.data.vertices)),
                                        fill_value=context.scene.vertex_angle_default)

    # At least testing with the bob duck mesh, using 2pi for the vertices worked just fine.
    # And it seems like a single island for the UV unwrapping is preferred to work fine.
    # Then, save the location of the cones into vertex_angles per Capouellez et al. 2023
    indices, angles = retrieve_cone_vertex_angles(
        context.scene.vertex_angles)
    vertex_angles[indices] = angles

    return vertex_angles


def save_vertex_angles(filepath: str | pathlib.Path, vertex_angles: np.ndarray) -> None:
    """ 
    Given NumPy array of vertex angles, save to filepath
    """
    # Ensures that we have all the QOL that pathlib.Path provides
    if filepath is not pathlib.Path:
        filepath = pathlib.Path(filepath)

    # Checks to see that parent exists
    if not filepath.parent.exists():
        raise OSError(f"Directory {filepath.parent} does not exist for {filepath.name}")

    # Makes sure that vertex angles is in the correct format
    if vertex_angles.ndim != 1:
        raise ValueError(
            f"Vertex angles is not 1D array! Instead is {vertex_angles.ndim}-dimensional array")

    np.savetxt(fname=filepath, X=vertex_angles, newline="\n")


class VertexAngleItem(bpy.types.PropertyGroup):
    """
    Custom data class to hold vertices and their assigned angles

    Useful sources below:
    https://sinestesia.co/blog/tutorials/using-uilists-in-blender/
    https://docs.blender.org/api/current/bpy.types.UIList.html
    https://blender.stackexchange.com/questions/15917/populate-a-list-with-custom-property-dictionary-data
    """

    # Properties to hold onto the data first.
    index: bpy.props.IntProperty(
        name='Vertex Index',
        description="Selected vertex index",
        default=0,
        min=0
    )

    # TODO: these item lists are only useful for certain mesh parametrization algorithms
    # i.e. Campen has ONLY angles per line, with each line matching a vertex index
    # Meanwhile, CEPS has to specify BOTH index AND angle for specified index
    # Meaning that each must have their own way of extracting from this list of vertices...
    # MEANING that THE CONCRETE implementation of the ABSTRACT CLASS must resolve how to handle info in this list...

    # TODO: ensure that the below is in RADIANS and not DEGREES
    # Which, the best way is to just not specify the subtyle to not
    # deal with such a headache.
    angle: bpy.props.FloatProperty(
        name="Vertex Angle",
        description="Specified angle constraint for selected vertex",
        default=0.0,
    )


class RANATO_UL_ItemList(bpy.types.UIList):
    """ UIList subclass. But is more concerned with the data stored in each property rather than
    how it is displayed.
    # TODO: this ultimately depends on WHAT strategy is being used...
    """

    def draw_item(self, context, layout, data, item, icon, _active_data, _active_propname, index) -> None:
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
            # layout.label(text=item.name, icon=custom_icon)
            row = layout.row()
            split = row.split(factor=0.01)
            split.row().label(text="", icon="DECORATE")
            split.row().prop(item, "index", text="Index", emboss=False)
            split.row().prop(item, "angle", text="Angle", emboss=False)
        elif self.layout_type in {"GRID"}:
            layout.alignment = "CENTER"
            layout.label(text=f"{item.index}", icon=custom_icon)


class LIST_OT_AddItem(bpy.types.Operator):
    """Add a new item to the list"""
    bl_idname = "vertex_angles.add_item"
    bl_label = "Add new item to list"
    bl_description = "Add vertex/angle entry to list"

    def execute(self, context) -> set[str]:
        # NOTE: .add() is inherited from bpy.props.CollectionProperty
        vertex_angles: bpy.props.CollectionProperty = context.scene.vertex_angles
        item: VertexAngleItem = vertex_angles.add()
        print(item)
        print(type(item))
        # TODO: ensure no duplicates...
        # And also add with default cone vertex angle

        # TODO: need to increment based on the index so far...
        item.index = len(vertex_angles) - 1
        item.angle = context.scene.cone_angle_default

        return {"FINISHED"}


class LIST_OT_RemoveItem(bpy.types.Operator):
    """
    Add a new item to the list

    Source:
    https://sinestesia.co/blog/tutorials/using-uilists-in-blender/
    """
    bl_idname = "vertex_angles.remove_item"
    bl_label = "Remove item from list"
    bl_description = "Remove vertex/angle entry from list"

    @classmethod
    def poll(cls, context):
        # TODO: find out why this is needed
        return context.scene.vertex_angles

    def execute(self, context) -> set[str]:
        # TODO: allow for redoing and undoing this remove item and add item!!!
        vertex_angles = context.scene.vertex_angles
        index = context.scene.list_index

        vertex_angles.remove(index)
        context.scene.list_index = min(
            max(0, index-1),
            len(vertex_angles) - 1
        )

        return {"FINISHED"}


class LIST_OT_Import(bpy.types.Operator, ImportHelper):
    """
    Imports vertex angles from file.

    Source:
    https://sinestesia.co/blog/tutorials/using-blenders-filebrowser-with-python/
    https://blender.stackexchange.com/questions/42654/ui-how-to-add-a-file-browser-to-a-panel
    """
    bl_idname = "vertex_angles.import"
    bl_label = "Import"
    bl_description = "Import formatted .txt of vertex angles"
    directory: bpy.props.StringProperty(subtype='DIR_PATH', options={'SKIP_SAVE', 'HIDDEN'})

    # FIXME: below are not properly filtering files (should only be either _Th_hat or .txt options)
    bl_file_extensions: str = "_Th_hat"
    filename_ext: str = "_Th_hat"
    filter_glob = bpy.props.StringProperty(default="*_Th_hat;*.txt", options={'HIDDEN'})
    one_indexed: bpy.props.BoolProperty(name="1-index to 0-index", default=True,
                                        description="Converts selected cones/vertex angle file from 1-indexed to 0-indexed format")

    @staticmethod
    def _import(filepath: str) -> np.ndarray:
        """
        Helper method reading from selected filepath
        """

        # Which is to say that it uses NumPy to read in the filepath
        # NOTE: just make it all float type then cast the left-most col to int...
        # TODO: might be better to have dtype as "object" instead of float so that we're preserving the mixed datatype in the file rather than having to cast float to int.
        index_angles = np.loadtxt(filepath, dtype=float, delimiter=" ")

        # File from CEPS, which has vertices with explicitly-labeled indices
        if index_angles.ndim == 2 and index_angles.shape[1] == 2:
            # TODO: check if this is correct....
            return index_angles
        # File from Campen et al. 2021, which does not have a column specifying indices with angles (is implicitly inferred)
        elif index_angles.ndim == 1:
            indices: np.ndarray = np.arange(index_angles.shape[0], dtype=float)
            return np.column_stack((indices, index_angles))
        else:
            raise ValueError("Index angles file not 1D or 2D")
        # Perform some adjustments...
        # If vertex angles are not explicitly specified, then go ahead and do that.

    def execute(self, context) -> set[str]:
        """ Executed after invoking import. 
        Which is to say that after the user selects a .txt of vertex angles, then this 
        performs the internal logic.
        """
        # Now that we have the filepath from ImportHelper
        vertex_angles: bpy.types.CollectionProperty = context.scene.vertex_angles
        indices_angles: np.ndarray = self._import(self.filepath)
        vertex_angles.clear()

        # Because the MovingCones locator has 1-indexed vertex indices, have option to adapt for
        # Campen algorithm
        for index, angle in indices_angles:
            item: VertexAngleItem = vertex_angles.add()
            item.index = int(index)

            if self.one_indexed:
                item.index -= 1
                assert item.index >= 0

            item.angle = angle

        return {"FINISHED"}

    def invoke(self, context, event):
        """ Sets default folder upon calling this operator to __temp__ folder.
        https://docs.blender.org/api/current/bpy.types.FileHandler.html
        """
        # Need to set directory in case user changes the directory in preferences
        preferences: bpy.types.AddonPreferences | None = bpy.context.preferences.addons[
            ADDON_ID].preferences
        self.directory = preferences.directory_temp
        return self.invoke_popup(context)


class LIST_OT_Export(bpy.types.Operator, ExportHelper):
    """
    Exports vertex angles to file.
    NOTE: assuming Campen et al 2021-format of vertex angles
    Meaning, file is named [.obj name]_Th_hat and has implicit vertex indicies
    """
    bl_idname = "vertex_angles.export"
    bl_label = "Export vertex angles"
    bl_description = "Export formatted .txt of vertex angles"
    filename_ext: str = "_Th_hat"
    filepath = bpy.props.StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        """ Executed after invoking import. 
        Which is to say that after the user selects a .txt of vertex angles, then this 
        performs the internal logic.

        Args:
            context (_type_): TODO

        Returns:
            set[str]: TODO
        """
        # TODO: need to process through the list of vertex angles...
        # Inherit from Campen
        filepath = pathlib.Path(self.filepath)
        save_vertex_angles(context,
                           directory_temp=filepath.parent.as_posix(),
                           temp_filename=filepath.name)

        return {"FINISHED"}


class LIST_OT_Clear(bpy.types.Operator):
    """Clears list of cone angles"""
    bl_idname = "vertex_angles.clear"
    bl_label = ""
    bl_description = ""

    def execute(self, context) -> set[str]:
        # NOTE: .add() is inherited from bpy.props.CollectionProperty
        context.scene.vertex_angles.clear()

        return {"FINISHED"}


class LIST_OT_Apply(bpy.types.Operator):
    """Apply list of cone angles"""
    bl_idname = "vertex_angles.apply"
    bl_label = ""
    bl_description = "Apply default cone vertex angle to list of cone vertices"

    def execute(self, context) -> set[str]:
        # NOTE: .add() is inherited from bpy.props.CollectionProperty
        vertex_angles = context.scene.vertex_angles
        cone_angle_default = context.scene.cone_angle_default
        for vertex in vertex_angles:
            vertex.angle = cone_angle_default

        return {"FINISHED"}
