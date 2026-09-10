"""
Entry point for Blender add-on
"""
import math

import bpy.props
import bpy.types
import bpy.utils

from .panels import (RANATO_MT_ExportAngles, RANATO_PT_generate_contours,
                     RANATO_PT_intersection_settings,
                     RANATO_PT_invisibility_settings, RANATO_PT_main,
                     RANATO_PT_mesh_export, RANATO_PT_optimization_settings,
                     RANATO_PT_uv_unwrap, RANATO_PT_vertex_angles)
from .pipeline.export_mesh import RANATO_OT_Export_Mesh
from .pipeline.generate_contours.generate_contours_main import \
    RANATO_OT_pipeline
from .pipeline.generate_contours.generate_contours_settings import (
    IntersectionSettings, InvisibilitySettings, OptimizationSettings)
from .pipeline.locate_cones import RANATO_OT_locate_cones
from .pipeline.search_mesh import RANATO_OT_search_mesh_operator
from .pipeline.uv_unwrap.uv_unwrap_main import RANATO_OT_uv_unwrap
from .pipeline.uv_unwrap.uv_unwrap_settings import (BFFSettings,
                                                    CampenSettings,
                                                    CEPSSettings, CETMSettings,
                                                    UVUnwrapperSelection)
from .pipeline.vertex_angles import (LIST_OT_AddItem, LIST_OT_Apply,
                                     LIST_OT_Clear, LIST_OT_Export,
                                     LIST_OT_Import, LIST_OT_RemoveItem,
                                     RANATO_UL_ItemList, VertexAngleItem)
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

    RANATO_MT_ExportAngles,

    RANATO_PT_main,
    RANATO_PT_mesh_export,
    RANATO_PT_vertex_angles,
    RANATO_OT_locate_cones,

    # Lists
    VertexAngleItem,  # TODO: rename
    RANATO_UL_ItemList,
    LIST_OT_AddItem,
    LIST_OT_RemoveItem,
    LIST_OT_Import,
    LIST_OT_Export,
    LIST_OT_Clear,
    LIST_OT_Apply,


    # UV Unwrapping
    CampenSettings,  # TODO: rename so clear that it's Blender associated stuff
    CEPSSettings,  # TODO: rename so clear that it's Blender associated stuff
    CETMSettings,  # TODO: rename so clear that it's Blender associated stuff
    BFFSettings,  # TODO: rename so clear that it's Blender associated stuff
    UVUnwrapperSelection,  # TODO: rename so clear it's blender thing
    RANATO_PT_uv_unwrap,

    # Contour Generation
    RANATO_PT_generate_contours,
    RANATO_PT_optimization_settings,
    RANATO_PT_intersection_settings,
    RANATO_PT_invisibility_settings,
    # TODO: have subpanels here as well.
    OptimizationSettings,
    IntersectionSettings,
    InvisibilitySettings
]


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)

    # TODO: for these below, make getter functions????
    bpy.types.Scene.vertex_angles = bpy.props.CollectionProperty(type=VertexAngleItem)
    bpy.types.Scene.vertex_angle_default = bpy.props.FloatProperty(
        name="Default Vertex Angle (radians)", default=math.pi * 2.0)
    bpy.types.Scene.cone_angle_default = bpy.props.FloatProperty(
        name="Default Cone Angle (radians)", default=6.0)
    bpy.types.Scene.list_index = bpy.props.IntProperty(
        name="Index for vertex_angles", default=0)
    bpy.types.Scene.target_mesh = bpy.props.PointerProperty(
        name="Select Mesh", type=bpy.types.Object)
    bpy.types.Scene.uv_unwrap_settings = bpy.props.PointerProperty(type=UVUnwrapperSelection)
    bpy.types.Scene.locate_cones_distortion = bpy.props.FloatProperty(
        name="Distortion", default=0.2)
    bpy.types.Scene.optimization_settings = bpy.props.PointerProperty(type=OptimizationSettings)
    bpy.types.Scene.intersection_settings = bpy.props.PointerProperty(type=IntersectionSettings)
    bpy.types.Scene.invisibility_settings = bpy.props.PointerProperty(type=InvisibilitySettings)
    bpy.types.Scene.svg_output_mode = bpy.props.EnumProperty(
        name="SVG Output Mode",
        items=[("1", "UNIFORM_SEGMENTS",           "All contours in uniform color"),
               ("2", "UNIFORM_VISIBLE_SEGMENTS",   "Visible contours in uniform color"),
               ("3", "CONTRAST_INVISIBLE_SEGMENTS", "Visible and invisible segments in a different color"),
               ("4", "RANDOM_CHAINS",         "Chains in random colors"),
               ("5", "UNIFORM_CHAINS",        "Chains in uniform color"),
               ("6", "UNIFORM_VISIBLE_CHAINS", "Visible chains in uniform color"),
               ("7", "UNIFORM_VISIBLE_CURVES", "Visible curves with no breaks at special points"),
               ("8", "UNIFORM_CLOSED_CURVES", "All closed curves with no breaks at special points"),
               ("9", "UNIFORM_SIMPLIFIED_VISIBLE_CURVES",
                "All visible closed curves with simplification"),
               ],
        default="2",
        description="Method for computing quantitative visibility"
    )


def unregister() -> None:
    del bpy.types.Scene.vertex_angles
    del bpy.types.Scene.vertex_angle_default
    del bpy.types.Scene.cone_angle_default
    del bpy.types.Scene.list_index   # TODO: rename to active_index or something
    del bpy.types.Scene.target_mesh
    del bpy.types.Scene.uv_unwrap_settings
    del bpy.types.Scene.locate_cones_distortion
    del bpy.types.Scene.optimization_settings
    del bpy.types.Scene.intersection_settings
    del bpy.types.Scene.invisibility_settings
    del bpy.types.Scene.svg_output_mode

    for cls in reversed(classes):
        # TODO: unregister reversed?
        bpy.utils.unregister_class(cls)
