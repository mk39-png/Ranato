from .operators import operators
from . import render
from . import panels
import bpy

bl_info = {
    "name": "Ranato",
    "description": "Blender implementation of Algebraic Smooth Occluding Contours paper",
    "author": "Kevin Ha",
    "version": (0, 0, 1, 'Alpha'),
    "blender": (4, 4, 0),
    "category": "Render"
}

#  register operators, panels, menu items, etc


def register():
    panels.register()
    render.register()
    operators.register()


def unregister():
    panels.unregister()
    render.unregister()
    operators.unregister()


# This allows you to run the script directly from Blender's Text editor
# to test the add-on without having to install it.
if __name__ == "__main__":
    register()

    # Seems like we have to execute the panels the viewport render portion...
