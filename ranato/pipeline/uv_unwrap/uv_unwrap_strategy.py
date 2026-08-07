"""
Holds strategy pattern for UV unwrapping (aka mesh parametrization/conformal equivalence) algorithms
"""

from abc import ABC, abstractmethod
from typing import Any

from bpy.types import Context, PointerProperty, PropertyGroup, UILayout


class UVUnwrapStrategy(ABC):
    """ 
    Abstract class providing structure for any mesh parametrization (UV unwrapping) algorithm 
    utilizing a strategy pattern.
    """
    _id: str = ""  # used for accessing strategy attribute from UVUnwrapperSettings for displaying options from dropdown
    bl_idname: str = ""
    bl_label: str = ""
    # directory_temp: str = ""

    @abstractmethod
    def _call_uv_unwrapper(self, args) -> Any:
        """ 
        Internal helper function that calls external script if needed.

        Raises:
            NotImplementedError: method not implemented
        """
        raise NotImplementedError

    def draw(self, layout: UILayout, settings: PointerProperty) -> None:
        """
        Concrete method for Blender panels drawing only relevant fields to the UV unwrapping algorithm.

        Args:
            layout (UILayout): UI layout referenced
            settings (PointerProperty): settings to display in UI
        """
        # e.g. cetm, campen, ceps, bff
        base_attribute: PointerProperty = getattr(settings, self._id)
        attributes: list[str] = [
            name for name in dir(base_attribute) if name.startswith("prop")
        ]

        # print("Base attribute: ", base_attribute)
        # print("Attributes: ", attributes)

        for x in attributes:
            layout.prop(base_attribute, x)

    @abstractmethod
    def execute(self, context: Context, settings: PointerProperty) -> None:
        """ 
        Method for Blender operations executing actual UV unwrapping.

        Args:
            context (Context): Blender context variables to utilize
            settings (PointerProperty): settings to utilize for UV unwrapping execution
        """
        raise NotImplementedError
