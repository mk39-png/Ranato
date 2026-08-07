"""
UI elements of Ranato
"""

import bpy
import bpy.props
import bpy.types

from .pipeline.uv_unwrap.bff import BFFStrategy
from .pipeline.uv_unwrap.campen import CampenStrategy
from .pipeline.uv_unwrap.ceps import CEPSStrategy
from .pipeline.uv_unwrap.cetm import CETMSettings, CETMStrategy
from .pipeline.uv_unwrap.uv_unwrap_main import STRATEGIES


class RANATO_PT_mesh_export(bpy.types.Panel):
    """ Exporting meshes.
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

        # DEPRECATED: old mesh exporting is below:
        # layout.label(text="Export Mesh")
        # row: bpy.types.UILayout = layout.row(align=True)
        # row.operator(operator="object.search_mesh_operator",
        #  text = "Search Mesh", icon = "COLLAPSEMENU")

        row = layout.row()
        row.prop(data=context.scene, property="target_mesh", emboss=True)
        row = layout.row()
        row.enabled = hasattr(
            context.scene, "target_mesh") and context.scene.target_mesh is not None
        row.operator(operator="object.export_mesh_operator",
                     text="Export Mesh", icon="COLLAPSEMENU")


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

        # --- UV UNWRAPPER ---

        # TODO: need to get parametrizations fo rthis!
        row: bpy.types.UILayout = layout.row(align=True)
        # layout.separator()
        # sub_row: bpy.types.UILayout = row.column_flow()
        # sub_row = row.box()
        # row = layout.row()

        # https://projects.blender.org/blender/blender/src/commit/2d8a95775148e00e07d8aca587ec5faecbe44c24/scripts/startup/bl_ui/properties_view_layer.py
        # TODO: below is shared by all UV unwrapping algorithms (if applicable)
        # layout.use_property_split = True
        # layout.use_property_decorate = False

        # --- Vertex Angles Specifier ---
        row = layout.row()
        row.label(text="Default Vertex Angle (radians):")
        layout.prop(data=context.scene, property="vertex_angle_default", emboss=True)
        row = layout.row()
        row.label(text="Cone Vertices (overrides vertex angles of inputted vertices)")
        layout.label(text="Applicable to Campen et al. 2021 and CEPS algorithms.")

        row = layout.row()
        row.template_list("RANATO_UL_ItemList", "ranato_list",
                          scene, "vertex_angles",
                          scene, "list_index")
        col = row.column()
        col.operator("vertex_angles.add_item", icon="ADD", text="")
        col.operator("vertex_angles.remove_item", icon="REMOVE", text="")
        col.operator("vertex_angles.sort_item", icon="COLLAPSEMENU", text="")
        row = layout.row()

        # print(scene.vertex_angles[0].vertex_index)
        # print(scene.vertex_angles[0].angle)
        # TODO: have option to specify default angle for vertices...
        # TODO: option to specify default scale for vertices...

        if scene.list_index >= 0 and scene.vertex_angles:
            item = scene.vertex_angles[scene.list_index]
            row = layout.row()
            row.prop(item, "vertex_index")
            row.prop(item, "angle")


class RANATO_PT_uv_unwrap(bpy.types.Panel):
    """ Calls UV unwrapper
    """
    bl_label = "UV Unwrap (Mesh Parametrization)"
    bl_idname = "panel.ranato_uv_unwrap"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    bl_parent_id = "panel.ranato_main"

    def draw(self, context) -> None:
        layout: bpy.types.UILayout | None = self.layout
        scene: bpy.types.Scene | None = context.scene
        row: bpy.types.UILayout = layout.row()

        # Retrieve what UV unwrapper we have right now!
        settings = scene.uv_unwrap_settings
        # print(settings)

        # HACK: assume campen strategy for now
        strategy: CampenStrategy | CEPSStrategy | CETMSettings | BFFStrategy = STRATEGIES[
            settings.method]
        # print(strategy)

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
    bl_idname = "panel.ranato_generator_contours"
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

    # TODO: include button to clear cache, which removes files from __temp__

    def draw(self, context) -> None:
        layout: bpy.types.UILayout | None = self.layout

        if layout is None:
            assert ("Layout should not be None when drawing!")

        # --- Invoke search popup for user to select mesh ---
        # row: bpy.types.UILayout = layout.row(align=True)
        # layout.separator()
        # layout.label(text="Export Mesh")
        # row: bpy.types.UILayout = layout.row(align=True)
        # row.operator(operator="object.search_mesh_operator",
        #  text = "Search Mesh", icon = "COLLAPSEMENU")

        # TODO: set active if UV unwrapping is detected for the correct mesh...
        # row.operator("render.render")
