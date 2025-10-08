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
    bl_label = "Ranato Settings"
    bl_idname = "RENDER_PT_layout"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"

    # def __del__(self) -> None:
    #     raise Exception("UNIMPLEMENTED")

    def draw(self, context) -> None:
        # Objects also include non-meshes

        # https://docs.blender.org/api/current/bpy.data.html
        # What if I only want the active meshes?
        # print(bpy.types.BlendDataMeshes(bpy.data.meshes[0]))
        # TODO: use "bpy_struct.is_property_hidden" on meshes to find out which mesh is visible...
        # Then for visible meshes... oh wait... we dont want to apply to ALL visible meshes

        layout: UILayout | None = self.layout
        scene: Scene | None = context.scene

        # This will show us if external files work OK
        # src.tester.testing()

        # TODO: Implement panel to UV unwrap using specialized algorithm since we do not
        #       want to unwrap every time we call the contour generator.

        # Select frame to render
        # Create an row where the buttons are aligned to each other.
        layout.label(text="Frames Render")
        row: UILayout = layout.row(align=True)
        row.prop(scene, "frame_start")
        row.prop(scene, "frame_end")

        # https://docs.blender.org/api/current/bpy.types.Operator.html#enum-search-popup
        # Invoke search popup for user to select mesh
        layout.label(text="Render:")
        row = layout.row()
        row.operator('object.search_mesh_operator', text="Search")

        layout.separator()

        # TODO: include button that separately calculates UV coordinates

        # Create two columns, by using a split layout.
        # split = layout.split()

        # First column
        # col = split.column()
        # col.label(text="Column One:")
        # col.prop(scene, "frame_end")
        # col.prop(scene, "frame_start")

        # Second column, aligned
        # col = split.column(align=True)
        # col.label(text="Column Two:")
        # col.prop(scene, "frame_start")
        # col.prop(scene, "frame_end")

        # Big render button
        layout.label(text="Render:")
        row = layout.row()
        row.scale_y = 2.0
        row.operator("render.render")

        # Different sizes in a row
        # layout.label(text="Different button sizes:")
        # row = layout.row(align=True)
        # row.operator("render.render")


classes: list[type[RanatoPanel]] = [RanatoPanel]

register, unregister = bpy.utils.register_classes_factory(classes)

if __name__ == "__main__":
    register()
