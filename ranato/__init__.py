"""
Entry point for Blender add-on
"""
import bpy

from .panels import PANEL_PT_ranato_panel
from .pipeline import OBJECT_OT_pipeline
from .preferences import OBJECT_OT_addon_preferences, RanatoPreferences
from .search_mesh import OBJECT_OT_search_mesh_operator

classes: list = [OBJECT_OT_search_mesh_operator,
                 OBJECT_OT_pipeline,
                 OBJECT_OT_addon_preferences,
                 PANEL_PT_ranato_panel,
                 RanatoPreferences,]

register, unregister = bpy.utils.register_classes_factory(classes)
