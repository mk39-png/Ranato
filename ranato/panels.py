import bpy
from bpy.types import Panel, Scene, UILayout

# https://blender.stackexchange.com/questions/202570/multi-files-to-addon
# Setting up addon with multiple files

# https://docs.blender.org/api/current/bpy.types.Panel.html

# Referencing below to see how Render Engine interacts with Panel
# (and changing it all so that the panel only appears when user activates Ranato)
# https://github.com/bnpr/Malt/blob/725f509ab25be736cb592cf1e9d5258ed4271e8a/BlenderMalt/MaltMaterial.py#L98


class RanatoPanel(Panel):
    """
    Creates a Panel in the scene context of the properties editor.
    """
    bl_label = "Ranato"
    bl_idname = "RENDER_PT_layout"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"

    def draw(self, context) -> None:
        layout: UILayout | None = self.layout
        scene: Scene | None = context.scene

        # TODO: Implement panel to UV unwrap using specialized algorithm since we do not
        #       want to unwrap every time we call the contour generator.

        # TODO: implement ability to render multiple frames
        # layout.label(text="Frames Render")
        # row: UILayout = layout.row(align=True)
        # row.prop(scene, "frame_start")
        # row.prop(scene, "frame_end")

        # https://docs.blender.org/api/current/bpy.types.Operator.html#enum-search-popup
        # Invoke search popup for user to select mesh
        layout.label(text="Select Mesh :")
        row: UILayout = layout.row()
        row.operator('object.search_mesh_operator', text="Select Mesh")

        layout.separator()

        # TODO: include button that separately calculates UV coordinates

        # Big render button
        # layout.label(text="Render:")
        row = layout.row()
        row.operator("render.render")
        row.scale_y = 2.0


classes: list[type[RanatoPanel]] = [RanatoPanel]

register, unregister = bpy.utils.register_classes_factory(classes)

if __name__ == "__main__":
    register()
