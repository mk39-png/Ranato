"""
UI elements of Ranato
"""

import bpy
import bpy.types

from .pipeline.uv_unwrap.bff import BFFStrategy
from .pipeline.uv_unwrap.campen import CampenStrategy
from .pipeline.uv_unwrap.ceps import CEPSStrategy
from .pipeline.uv_unwrap.cetm import CETMStrategy
from .pipeline.uv_unwrap.uv_unwrap_main import STRATEGIES
from .pipeline.uv_unwrap.uv_unwrap_settings import UVUnwrapperSelection


class RANATO_PT_mesh_export(bpy.types.Panel):
    """ Exporting meshes using Blender's built-in .obj exporter
    """
    bl_label = "Mesh Export"
    bl_idname = "panel.ranato_mesh_export"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    bl_parent_id = "panel.ranato_main"

    def draw(self, context) -> None:
        layout: bpy.types.UILayout | None = self.layout
        row: bpy.types.UILayout = layout.row(align=True)
        row.prop(data=context.scene, property="target_mesh", emboss=True)
        row = layout.row()
        row.enabled = hasattr(
            context.scene, "target_mesh") and context.scene.target_mesh is not None
        row.operator(operator="object.export_mesh_operator",
                     text="Export Mesh", icon="EXPORT")


class RANATO_MT_ExportAngles(bpy.types.Menu):
    bl_idname = "ranato.menu"
    bl_label = "Export Angles"

    def draw(self, context):
        layout = self.layout
        row = layout.row()


class RANATO_PT_vertex_angles(bpy.types.Panel):
    """ Specifying vertex angles.
    """
    bl_label = "Vertex Angles"
    bl_idname = "panel.ranato_vertex_angles"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    bl_parent_id = "panel.ranato_main"

    def draw(self, context) -> None:
        layout: bpy.types.UILayout | None = self.layout
        scene: bpy.types.Scene | None = context.scene

        # NOTE: referenced source below for implementation
        # https://projects.blender.org/blender/blender/src/commit/2d8a95775148e00e07d8aca587ec5faecbe44c24/scripts/startup/bl_ui/properties_view_layer.py

        # --- Vertex Angles Specifier ---
        row: bpy.types.UILayout = layout.row(align=True)
        row.label(text="Default Vertex Angle (radians):")
        layout.prop(data=context.scene, property="vertex_angle_default", emboss=True)
        row = layout.row()
        row.label(text="Cone Vertices (overrides vertex angles of inputted vertices)")

        layout.label(text="Applicable to Campen et al. 2021 algorithm.")

        row = layout.row()
        row.template_list("RANATO_UL_ItemList", "ranato_list",
                          scene, "vertex_angles",
                          scene, "list_index")
        col: bpy.types.UILayout = row.column()
        col.operator("vertex_angles.add_item",
                     icon="ADD", text="")
        col.operator("vertex_angles.remove_item",
                     icon="REMOVE", text="")
        col.operator("vertex_angles.import", icon="IMPORT", text="")

        row = layout.row()

        if scene.list_index >= 0 and scene.vertex_angles:
            item = scene.vertex_angles[scene.list_index]
            row = layout.row()
            row.prop(item, "index")
            row.prop(item, "angle")


class RANATO_PT_uv_unwrap(bpy.types.Panel):
    """ Calls UV unwrapper
    """
    bl_label = "UV Unwrap (Mesh Parametrization)"
    bl_idname = "RANATO_PT_uv_unwrap"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    bl_parent_id = "panel.ranato_main"

    def draw(self, context) -> None:
        layout: bpy.types.UILayout | None = self.layout
        scene: bpy.types.Scene | None = context.scene
        row: bpy.types.UILayout = layout.row()

        # Retrieve what UV unwrapper we have right now!
        settings: UVUnwrapperSelection = scene.uv_unwrap_settings

        # TODO: have button to reset arguments for chosen UV unwrapping method
        strategy: CampenStrategy | CEPSStrategy | CETMStrategy | BFFStrategy = STRATEGIES[
            settings.method]

        # TODO: have CANCEL button to stop the UV unwrapping execution

        # TODO: perhaps combine the strategy with settings...
        # layout.label(text="TODO: have button to restore to default parameters", icon="EXPORT")
        layout.prop(settings, "method")
        box: bpy.types.UILayout = layout.box()
        box.label(text=f"Settings for {strategy.bl_label}")
        strategy.draw(box, settings)

        # NOTE: this remains the same... calling this function that encapsulates whatever
        # UV unwrapping was selected
        row = layout.row()  # new row so that button is at the bottom
        row.operator(operator="object.uv_unwrap",
                     text="Generate Unwrapping", icon="UV")


class RANATO_PT_generate_contours(bpy.types.Panel):
    """ Generates algebraic contours
    """
    bl_label = "Generate Contours"
    bl_idname = "RANATO_PT_generate_contours"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    bl_parent_id = "panel.ranato_main"

    def draw(self, context) -> None:
        layout: bpy.types.UILayout | None = self.layout
        scene: bpy.types.Scene | None = context.scene
        row: bpy.types.UILayout = layout.row()

        # TODO: give options to select the following:
        # invisibility method
        # svg_mode
        # weight
        # trim
        # pad
        # show nodes

        row.operator("object.pipeline", text="Generate Contours", icon="LINCURVE")
        row.scale_y = 2.0


class RANATO_PT_main(bpy.types.Panel):
    """
    Creates a Panel in the scene context of the properties editor.
    """
    bl_label = "Ranato"
    bl_idname = "panel.ranato_main"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"

    def draw(self, context) -> None:
        layout: bpy.types.UILayout | None = self.layout

        if layout is None:
            assert ("Layout should not be None when drawing!")

        # --- Invoke search popup for user to select mesh ---
        # row = layout.row()
        # row: bpy.types.UILayout = layout.row(align=True)
        # TODO: include button to clear cache, which removes files from __temp__
        # row.operator(operator="object.search_mesh_operator",
        #  text="Search Mesh", icon="COLLAPSEMENU")
