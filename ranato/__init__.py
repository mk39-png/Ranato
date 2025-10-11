import os
import subprocess
import sys
from typing import Any

from . import operators, panels

bl_info: dict[str, Any] = {
    "name": "Ranato",
    "description": "Blender implementation of Algebraic Smooth Occluding Contours paper",
    "author": "Kevin Ha",
    "version": (0, 0, 1, 'Alpha'),
    "blender": (4, 5, 0),
    "category": "Render"
}


# https://blenderartists.org/t/can-i-install-pandas-or-other-modules-into-blenders-python/1375122
# NOTE: the below only works for Windows systems
try:
    import bpy
    import igl
    import matplotlib
    import mpmath as mp
except ImportError:
    # Blender Python interpreter location
    python_exe: str = os.path.join(sys.prefix, 'bin', 'python.exe')

    # This is the folder that Blender automatically holds its installed addons
    requirements_path: str = os.path.join(os.path.dirname(__file__), "requirements.txt")

    # Pip Upgrade
    subprocess.call([python_exe, "-m", "ensurepip"])
    subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])

    # Install Required Packages
    subprocess.call([python_exe, "-m", "pip", "install", "-r", requirements_path])


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
