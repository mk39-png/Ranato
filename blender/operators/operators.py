import bpy
from bpy.props import EnumProperty


# Resources helping with understanding what EnumProperty is all about
# (and also understanding how "register" interacts with the rest of the addon)
# https://blender.stackexchange.com/questions/247695/invoke-search-popup-for-a-simple-panel
# https://docs.blender.org/api/current/bpy.types.Operator.html#enum-search-popup
# TODO: code that gets all meshes in the scene collection


# TODO: change this to only get objects within the dependency graph....
# TODO: also, reclarify what operators are.
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

    # TODO: get the active camera as well...
    # TODO: get the UV coordinates?
    # TODO: get the vertex coordinates...

    # Well, I need the Vertex Position, Texture Coordinates, Corner Normals, Face indices into vertex posn vertex tex coord, vertex normal.
    # https://docs.blender.org/api/current/bpy.types.Depsgraph.html
    def execute(self, context):
        depsgraph = context.evaluated_depsgraph_get()

        for object_instance in depsgraph.object_instances:
            # This is an object which is being instanced.
            obj = object_instance.object
            # `is_instance` denotes whether the object is coming from instances (as an opposite of
            # being an emitting object. )
            if not object_instance.is_instance:
                print(f"Object {obj.name} at {object_instance.matrix_world}")
            else:
                # Instanced will additionally have fields like uv, random_id and others which are
                # specific for instances. See Python API for DepsgraphObjectInstance for details,
                print(
                    f"Instance of {obj.name} at {object_instance.matrix_world}")

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


if __name__ == "__main__":
    register()
