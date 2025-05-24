import bpy
from bpy.props import EnumProperty


# Resources helping with understanding what EnumProperty is all about 
# (and also understanding how "register" interacts with the rest of the addon)
# https://blender.stackexchange.com/questions/247695/invoke-search-popup-for-a-simple-panel
# https://docs.blender.org/api/current/bpy.types.Operator.html#enum-search-popup
# TODO: code that gets all meshes in the scene collection
def get_objects(self, context):
    enum = []

    for obj in bpy.data.collections["Collection"].all_objects:
        id_ = str(obj.name) 
        name = id_ 
        desc = "Description " + str(obj.name) 
        enum.append((id_, name, desc,))

    return enum


class SearchEnumOperator(bpy.types.Operator):
    bl_idname = "object.search_enum_operator"
    bl_label = "Search Enum Operator"
    bl_property = "my_search"

    # https://blenderartists.org/t/menu-enumproperty/1446897
    my_search: EnumProperty(items=get_objects)
 
    def execute(self, context):
        self.report({'INFO'}, "Selected:" + self.my_search)
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.invoke_search_popup(self)
        return {'RUNNING_MODAL'}


# Logistics functions
def register():
    bpy.utils.register_class(SearchEnumOperator)

def unregister():
    bpy.utils.unregister_class(SearchEnumOperator)


# // Do I really need this?
if __name__ == "__main__":
    register()
