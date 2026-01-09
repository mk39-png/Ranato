import bpy
import bpy.props
import bpy.types

# https://blender.stackexchange.com/questions/202570/multi-files-to-addon
# Setting up addon with multiple files

# https://docs.blender.org/api/current/bpy.types.Panel.html

# Referencing below to see how Render Engine interacts with Panel
# (and changing it all so that the panel only appears when user activates Ranato)
# https://github.com/bnpr/Malt/blob/725f509ab25be736cb592cf1e9d5258ed4271e8a/BlenderMalt/MaltMaterial.py#L98
# https://github.com/Griperis/BlenderDataVis/blob/master/data_vis/operators/surface_chart.py


class PANEL_PT_ranato_panel(bpy.types.Panel):
    """
    Creates a Panel in the scene context of the properties editor.
    """
    bl_label = "Ranato"
    bl_idname = "object.ranato_panel"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"

    # FIXME: have the option for the user to have the addon execute the uv unwrapper or not
    uv_unwrapper_toggle: bpy.props.BoolProperty(
        name="Toggle UV Unwrapping",
        description="Toggle for UV unwrapper process.",
        default=False
    )

    selected_mesh_name: bpy.props.StringProperty(
        name="n/a",
        default=f"n/a"
    )

    def draw(self, context) -> None:
        layout: bpy.types.UILayout | None = self.layout
        scene: bpy.types.Scene | None = context.scene

        if not layout:
            assert ("Layout should not be None when drawing!")

        # TODO: implement ability to render multiple frames
        layout.label(text="Frames Render")
        row: bpy.types.UILayout = layout.row(align=True)
        row.prop(scene, "frame_start")
        row.prop(scene, "frame_end")

        # TODO: Implement toggle to UV unwrap using specialized algorithm since we do not
        #       want to unwrap every time we call the contour generator.

        # https://docs.blender.org/api/current/bpy.types.Operator.html#enum-search-popup
        # Invoke search popup for user to select mesh
        layout.label(text="Select Mesh")
        row: bpy.types.UILayout = layout.row(align=True)
        # row.prop(self, "uv_unwrapper_toggle")
        row.operator("object.search_mesh_operator", text="Select Mesh", icon="COLLAPSEMENU")
        row.operator("object.pipeline", text="Generate Contours", icon="OUTLINER_DATA_VOLUME")

        # layout.operator("")
        layout.separator()

        # TODO: include a big button to generate occluding contours.
        #

        # Big render button
        layout.label(text="Render")
        row = layout.row()
        row.active = False

        # This operator should call an operator like "algebraic contours" or something.
        row.operator("render.render")
        row.scale_y = 2.0
