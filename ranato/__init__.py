"""
Entry point for Blender add-on
"""
import math

import bpy.props
import bpy.types
import bpy.utils

from .panels import (RANATO_PT_generate_contours, RANATO_PT_main,
                     RANATO_PT_mesh_export, RANATO_PT_uv_unwrap,
                     RANATO_PT_vertex_angles)
from .pipeline.generate_contours import RANATO_OT_pipeline
from .pipeline.search_mesh import (RANATO_OT_Export_Mesh,
                                   RANATO_OT_search_mesh_operator)
from .pipeline.uv_unwrap.bff import BFFSettings
from .pipeline.uv_unwrap.campen import CampenSettings
from .pipeline.uv_unwrap.ceps import CEPSSettings
from .pipeline.uv_unwrap.cetm import CETMSettings
from .pipeline.uv_unwrap.uv_unwrap_main import (RANATO_OT_uv_unwrap,
                                                UVUnwrapperSettings)
from .pipeline.vertex_angles import (LIST_OT_AddItem, LIST_OT_RemoveItem,
                                     LIST_OT_SortItem, RANATO_UL_ItemList,
                                     VertexAngleItem)
from .preferences import RANATO_OT_addon_preferences, RanatoPreferences

classes: list = [
    # NOTE: must register preferences first so that the other classes
    # can reference it
    RanatoPreferences,  # TODO: rename so clear it's blender thing
    RANATO_OT_search_mesh_operator,
    RANATO_OT_pipeline,
    RANATO_OT_addon_preferences,
    RANATO_OT_uv_unwrap,
    RANATO_OT_Export_Mesh,

    RANATO_PT_main,
    RANATO_PT_mesh_export,
    RANATO_PT_vertex_angles,

    # Lists
    VertexAngleItem,  # TODO: rename
    RANATO_UL_ItemList,
    LIST_OT_AddItem,
    LIST_OT_RemoveItem,
    LIST_OT_SortItem,

    # UV Unwrapping
    CampenSettings,  # TODO: rename so clear that it's Blender associated stuff
    CEPSSettings,  # TODO: rename so clear that it's Blender associated stuff
    CETMSettings,  # TODO: rename so clear that it's Blender associated stuff
    BFFSettings,  # TODO: rename so clear that it's Blender associated stuff
    UVUnwrapperSettings,  # TODO: rename so clear it's blender thing
    RANATO_PT_uv_unwrap,

    # Contour Generation
    RANATO_PT_generate_contours,
]


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.vertex_angles = bpy.props.CollectionProperty(type=VertexAngleItem)
    bpy.types.Scene.vertex_angle_default = bpy.props.FloatProperty(
        name="Angle (radians)", default=math.pi * 2.0)
    bpy.types.Scene.list_index = bpy.props.IntProperty(name="Index for vertex_angles", default=0)
    bpy.types.Scene.target_mesh = bpy.props.PointerProperty(
        name="Select Mesh", type=bpy.types.Object)
    bpy.types.Scene.uv_unwrap_settings = bpy.props.PointerProperty(type=UVUnwrapperSettings)


def unregister() -> None:
    del bpy.types.Scene.vertex_angles
    del bpy.types.Scene.vertex_angle_default
    del bpy.types.Scene.list_index   # TODO: rename to active_index or something
    del bpy.types.Scene.target_mesh
    del bpy.types.Scene.uv_unwrap_settings

    for cls in reversed(classes):
        # TODO: unregister reversed?
        bpy.utils.unregister_class(cls)


# register, unregister = bpy.utils.register_classes_factory(classes)
