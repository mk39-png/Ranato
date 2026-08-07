from bpy.props import BoolProperty
from bpy.types import PropertyGroup

from .uv_unwrap_strategy import UVUnwrapStrategy


class BFFSettings(PropertyGroup):
    """
    Settings for CETM and BFF conformal mapping methods.
    """
    # select CETM
    # select BFF

    # TODO: check UML references if such a boolean variable is used
    select: BoolProperty(
        name="CETM/BFF",
        description="Select CETM (true)/BFF (false)",
        default=True
    )


class BFFStrategy(UVUnwrapStrategy):
    """_summary_

    Args:
        UVUnwrapStrategy (_type_): _description_
    """
    _id = "bff"
    bl_idname = "BFF"
    bl_label = "BFF"

    def _call_uv_unwrapper(self):
        """
        """

    def draw(self, layout, settings):
        print()

    def execute(self, context, settings):
        """_summary_
        """
