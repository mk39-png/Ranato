"""
Global common.py to store state and information that is global to the Ranato addon.
(e.g. filepaths)

"""
import os
import sys

# Addon paths
DIRECTORY_BASE_ADDON: str = os.path.dirname(__file__)
DIRECTORY_TEMP: str = os.path.join(DIRECTORY_BASE_ADDON, "temp")

# UV unwrapper paths
DIRECTORY_UV_UNWRAPPER: str = os.path.abspath(os.path.join(DIRECTORY_BASE_ADDON,
                                                           "uv_unwrapper"))
FILEPATH_UV_UNWRAPPER: str = os.path.join(DIRECTORY_UV_UNWRAPPER, "script_conformal.py")

# Algebraic contours paths
DIRECTORY_ALGEBRAIC_CONTOURS: str = os.path.abspath(os.path.join(DIRECTORY_BASE_ADDON,
                                                                 "algebraic_contours"))

# Misc paths
DIRECTORY_BLENDER_PYTHON_ENV: str = sys.prefix
