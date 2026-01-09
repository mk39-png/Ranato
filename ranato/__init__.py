"""
Entry point for Blender add-on
"""
import bpy

from .panels import PANEL_PT_ranato_panel
from .pipeline import OBJECT_OT_pipeline
from .preferences import OBJECT_OT_addon_preferences, RanatoPreferences
from .search_mesh import OBJECT_OT_search_mesh_operator

# try:
#     import bpy
#     import cholespy
#     import igl
#     import matplotlib
#     import mpmath as mp
#     import pyalgcon
# except ImportError:
#     raise Exception("Failed to import packages")

# # Blender Python interpreter location
# python_exe: str = os.path.join(sys.prefix, 'bin', 'python.exe')

# # FIXME: fix the whole install of cholespy... which is not working for some reason...

# # This is the folder that Blender automatically holds its installed addons
# requirements_path: str = os.path.join(os.path.dirname(__file__), "requirements.txt")

# # Pip Upgrade
# subprocess.call([python_exe, "-m", "ensurepip"])
# subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])

# # Install Required Packages
# subprocess.call([python_exe, "-m", "pip", "install", "-r", requirements_path])


classes: list = [OBJECT_OT_search_mesh_operator,
                 OBJECT_OT_pipeline,
                 OBJECT_OT_addon_preferences,
                 PANEL_PT_ranato_panel,
                 RanatoPreferences,]

register, unregister = bpy.utils.register_classes_factory(classes)
