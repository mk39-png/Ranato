import bpy.props
import bpy.types
import numpy as np
from bpy_extras.io_utils import ImportHelper

# TODO: give option to load in vertex angles via file...


def retrieve_cone_vertex_angles(vertex_angles: bpy.types.bpy_prop_collection_idprop, ceps_format: bool = False):
    """_summary_

    Args:
        vertex_angles (bpy.types.bpy_prop_collection): _description_

    Returns:
        _type_: _description_
    """
    indices = np.fromiter((vertex.index for vertex in vertex_angles), dtype=int)
    angles = np.fromiter((vertex.angle for vertex in vertex_angles), dtype=float)
    return (indices, angles)


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
        item: VertexAngleItem = context.scene.vertex_angles.add()
        print(item)
        print(type(item))
        # TODO: ensure no duplicates...
        # And also add with default cone vertex angle

        # TODO: need to increment based on the index so far...
        item.index = 0
        item.angle = 0.0

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

    filepath = bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob = bpy.props.StringProperty(default='*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp',
                                           options={'HIDDEN'})

    @staticmethod
    def _import(filepath: str) -> np.ndarray:
        """
        Helper method reading from selected filepath
        """

        # Which is to say that it uses NumPy to read in the filepath
        # NOTE: just make it all float type then cast the left-most col to int...
        # TODO: might be better to have dtype as "object" instead of float so that we're preserving the mixed datatype in the file rather than having to cast float to int.
        index_angles = np.loadtxt(filepath, dtype=float, delimiter=" ")

        if index_angles.ndim == 2 and index_angles.shape[1] == 2:
            # TODO: check if this is correct....
            return index_angles
        elif index_angles.ndim == 1:
            indices: np.ndarray = np.arange(index_angles.shape[0], dtype=float)
            return np.column_stack((indices, index_angles))
        else:
            raise ValueError("Index angles file not 1D or 2D")
        # Perform some adjustments...
        # If vertex angles are not explicitly specified, then go ahead and do that.

    # def draw(self, context) -> None:
    #     layout: bpy.types.UILayout | None = self.layout
    #     scene: bpy.types.Scene | None = context.scene
    #     # layout.label(text="TODO: have button to restore to default parameters", icon="EXPORT")

    def execute(self, context) -> set[str]:
        """ Executed after invoking import. 
        Which is to say that after the user selects a .txt of vertex angles, then this 
        performs the internal logic.

        Args:
            context (_type_): TODO

        Returns:
            set[str]: TODO
        """

        # Now that we have the filepath from ImportHelper
        vertex_angles: bpy.types.CollectionProperty = context.scene.vertex_angles
        indices_angles: np.ndarray = self._import(self.filepath)
        vertex_angles.clear()

        for index, angle in indices_angles:
            item: VertexAngleItem = vertex_angles.add()
            item.index = int(index)
            item.angle = angle

        return {"FINISHED"}
