"""
Global common.py to store state and information that is global to the Ranato addon.
(e.g. filepaths)

"""
import pathlib
import sys

# Addon paths
DIRECTORY_BASE_ADDON: pathlib.Path = pathlib.Path(__file__).parent
DIRECTORY_TEMP: pathlib.Path = DIRECTORY_BASE_ADDON / "temp"

# UV unwrapper paths
DIRECTORY_UV_UNWRAPPER: pathlib.Path = DIRECTORY_BASE_ADDON / "uv_unwrapper"
FILEPATH_UV_UNWRAPPER: pathlib.Path = DIRECTORY_UV_UNWRAPPER / "script_conformal.py"

# Algebraic contours paths
DIRECTORY_ALGEBRAIC_CONTOURS: pathlib.Path = DIRECTORY_BASE_ADDON / "algebraic_contours"

# Misc paths
DIRECTORY_BLENDER_PYTHON_ENV: str = sys.prefix
