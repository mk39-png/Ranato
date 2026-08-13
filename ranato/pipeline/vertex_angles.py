import bpy.props
import bpy.types
# TODO: give option to load in vertex angles via file...

import numpy as np


def retrieveVertexAngles(vertex_angles: bpy.types.bpy_prop_collection):
    """_summary_

    Args:
        vertex_angles (bpy.types.bpy_prop_collection): _description_

    Returns:
        _type_: _description_
    """
    # Why is this needed? Well, to retrieve values from Blender rather than accessing Blender data directly...
    print(
        np.fromiter((vertex.index for vertex in vertex_angles), dtype=int)
    )
    print(
        np.fromiter((vertex.angle for vertex in vertex_angles), dtype=float)
    )


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
        name="Vertex Angle",  # TODO: rename to Vertex Angle
        description="Specified angle constraint for selected vertex",
        default=0.0,
    )


class RANATO_UL_ItemList(bpy.types.UIList):
    """ UIList subclass. But is more concerned with the data stored in each property rather than
    how it is displayed.
    # TODO: this ultimately depends on WHAT strategy is being used...
    """

    def draw_item(self, _context, layout, _data, item, icon, _active_data, _active_propname, index) -> None:
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

            # layout.prop(item, "index", text="Vertex Index", emboss=False)
            # layout.prop(item, "angle", text="Vertex Angle", emboss=False)
        elif self.layout_type in {"GRID"}:
            layout.alignment = "CENTER"
            layout.label(text=f"{item.index}", icon=custom_icon)


class LIST_OT_AddItem(bpy.types.Operator):
    """Add a new item to the list"""
    bl_idname = "vertex_angles.add_item"
    bl_label = "Add new item to list"
    bl_description = "Add vertex/angle entry to list"

    def execute(self, context) -> set[str]:
        item = context.scene.vertex_angles.add()

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


class LIST_OT_SortItem(bpy.types.Operator):
    """
    Add a new item to the list

    Source:
    https://sinestesia.co/blog/tutorials/using-uilists-in-blender/
    """
    bl_idname = "vertex_angles.sort_item"
    bl_label = "Move new item in list"
    bl_description = "Move vertex/angle entry in list"

    @classmethod
    def poll(cls, context):
        # TODO: find out why this is needed
        return context.scene.vertex_angles

    def execute(self, context) -> set[str]:
        vertex_angles = context.scene.vertex_angles
        index = context.scene.list_index

        if index > 0:
            vertex_angles.move(index, index - 1)
        # vertex_angles.move(0, 1)
        return {"FINISHED"}
