"""
Entry point for Blender add-on
"""
import bpy.props
import bpy.types
import bpy.utils

from .panels import LIST_OT_NewItem, PANEL_PT_Ranato
from .pipeline.generate_contours import OBJECT_OT_pipeline
from .pipeline.search_mesh import OBJECT_OT_search_mesh_operator
from .pipeline.uv_unwrap import (OBJECT_OT_uv_unwrap, RANATO_UL_ItemList,
                                 VertexAngleItem)
from .preferences import OBJECT_OT_addon_preferences, RanatoPreferences

classes: list = [
    # NOTE: must register preferences first so that the other classes
    # can reference it
    RanatoPreferences,
    OBJECT_OT_search_mesh_operator,
    OBJECT_OT_pipeline,
    OBJECT_OT_addon_preferences,
    OBJECT_OT_uv_unwrap,
    PANEL_PT_Ranato,
    VertexAngleItem,
    LIST_OT_NewItem,
    RANATO_UL_ItemList,
]


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.my_list = bpy.props.CollectionProperty(type=VertexAngleItem)
    # Shows which item to be highlighted???
    bpy.types.Scene.list_index = bpy.props.IntProperty(name="Index for my_list", default=0)
    bpy.types.Scene.target = bpy.props.PointerProperty(name="Select Mesh", type=bpy.types.Object)


def unregister() -> None:
    del bpy.types.Scene.my_list
    del bpy.types.Scene.list_index

    for cls in classes:
        bpy.utils.unregister_class(cls)


# register, unregister = bpy.utils.register_classes_factory(classes)
