"""
Global common.py to store state and information that is global to the Ranato addon.
(e.g. filepaths)

"""

# NOTE: this is goind to be bl_ext.vscode_development.ranato as long as common.py is in root directory of ranato/
ADDON_ID: str = __package__
DEBUG: bool = False


# TODO: target filenames for now...
INPUT_OBJ_FILENAME = "temp.obj"
OUTPUT_OBJ_FILENAME = "temp_out.obj"
