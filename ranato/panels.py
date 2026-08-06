import bpy
import bpy.props
import bpy.types


class LIST_OT_NewItem(bpy.types.Operator):
    """Add a new item to the list"""
    bl_idname = "my_list.new_item"
    bl_label = "Add new item to list"
    bl_description = "Add vertex/angle entry to list"

    def execute(self, context) -> set[str]:
        item = context.scene.my_list.add()
        item.vertex_index = 0
        item.angle = 0.0

        return {"FINISHED"}


class PANEL_PT_Ranato(bpy.types.Panel):
    """
    Creates a Panel in the scene context of the properties editor.
    """
    bl_label = "Ranato"
    bl_idname = "object.ranato_panel"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"

    selected_mesh_name: bpy.props.StringProperty(
        name="n/a",
        default=f"n/a"
    )

    def draw(self, context) -> None:
        layout: bpy.types.UILayout | None = self.layout
        scene: bpy.types.Scene | None = context.scene
        obj: bpy.types.Object | None = context.object

        # TODO: instead of storing this all like so...
        # Store it within a separate class that DOES NOT MODIFY the scene and add unnecessary attributes to it.
        # Makes everything a LOT clearer
        # if obj is not None:
        #     if not hasattr(obj, "my_list"):
        #         obj.my_list = bpy.props.CollectionProperty(type=VertexAngleItem)
        #     if not hasattr(obj, "list_index"):
        #         obj.list_index = bpy.props.IntProperty(name="Index for my_list", default=0)

        if layout is None:
            assert ("Layout should not be None when drawing!")

        # TODO: implement ability to render multiple frames
        # layout.label(text="Frames Render")
        row: bpy.types.UILayout = layout.row(align=True)

        # sub_row = row.menu_pie()
        # sub_row.label(text="Frame rendering not implemented")
        # sub_row.prop(scene, "frame_start")
        # sub_row.prop(scene, "frame_end")

        # https://docs.blender.org/api/current/bpy.types.Operator.html#enum-search-popup
        # Invoke search popup for user to select mesh
        # layout.label(text="Export Mesh")
        row: bpy.types.UILayout = layout.row(align=True)
        # row.prop(self, "uv_unwrapper_toggle")

        # TODO: change name to tat of the mesh
        row.operator(operator="object.search_mesh_operator",
                     text="Select Mesh", icon="COLLAPSEMENU")

        row = layout.row()
        row.prop(data=context.scene, property="target", emboss=True)
        # row.prop(context.scene, "target", context.scene,
        #  "objects", text="Select Object")
        print(bpy.types.Scene.target)

        layout.separator()
        layout.label(text="UV Unwrapper")

        # TODO: need to get parametrizations fo rthis!
        row: bpy.types.UILayout = layout.row(align=True)
        # TODO: put all the settings associated with uv unwrapping...
        sub_row: bpy.types.UILayout = row.column_flow()
        sub_row = row.box()
        row = layout.row()
        row.template_list("RANATO_UL_ItemList", "ranato_list",
                          scene, "my_list",
                          scene, "list_index")
        row = layout.row()
        row.operator("my_list.new_item", text="NEW")

        sub_row.operator(operator="object.uv_unwrap",
                         text="Generate Unwrapping", icon="UV")

        # --- Big render button ---
        layout.separator()
        layout.label(text="Render")
        row = layout.row()
        # TODO: set active if UV unwrapping is detected for the correct mesh...
        # row.active = False
        # row.operator("render.render")
        row.operator("object.pipeline", text="Generate Contours", icon="LINCURVE")
        row.scale_y = 2.0

        # --- Vertex Angles Specifier ---

        # if scene.list_index >= 0 and scene.my_list:
        #     item = scene.my_list[scene.list_index]
        #     row = layout.row()
        #     row.prop(item, "index")
        #     row.prop(item, "value")
