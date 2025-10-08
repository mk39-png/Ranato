from typing import Any

import bpy

from . import operators, panels

bl_info: dict[str, Any] = {
    "name": "Ranato",
    "description": "Blender implementation of Algebraic Smooth Occluding Contours paper",
    "author": "Kevin Ha",
    "version": (0, 0, 1, 'Alpha'),
    "blender": (4, 5, 0),
    "category": "Render"
}


# TODO: include a function that installs dependencies


def register() -> None:
    """
    Registering files associated with the addon for Blender to recognize.
    Called whenever addon is activated.
    """
    panels.register()
    operators.register()


def unregister() -> None:
    """
    Unregistering files associated with the addon for Blender to recognize.
    Called whenever addon is deactivated.
    """
    panels.unregister()
    operators.unregister()


# This allows you to run the script directly from Blender's Text editor
# to test the add-on without having to install it.
if __name__ == "__main__":
    register()
