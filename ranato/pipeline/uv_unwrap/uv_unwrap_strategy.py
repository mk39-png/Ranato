"""
Holds strategy pattern for UV unwrapping (aka mesh parametrization/conformal equivalence) algorithms
"""

import pathlib
from abc import ABC, abstractmethod
from typing import Any

import bpy
import numpy as np
from bpy.types import Context, Object, PointerProperty, PropertyGroup, UILayout

from ...common import ADDON_ID


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

    def _retrieve_vertex_angles(self, context: Context):
        """ Retrieves vertex angles for each index of selected mesh based on 
        default vertex angle and specified cone vertices.

        Args:
            context (Context): _description_
        """

        # TODO: ASSERT THAT MESH HAS BEEN SELECTED
        # TODO: check if it also has attribute as well...
        if context.scene.target_mesh is None:
            print("BAD NO MESH")
            return

        #
        # PREPARING FOR UV UNWRAPPING
        #
        selected_object: Object = context.scene.target_mesh
        directory_temp: str = pathlib.Path(
            bpy.context.preferences.addons[ADDON_ID].preferences.directory_temp)

        # After running the executable for locating cone indices, be sure to save where they are.
        # Construct an array of size matching number of vertices
        vertex_angles: np.ndarray = np.full(shape=(len(selected_object.data.vertices)),
                                            fill_value=context.scene.vertex_angle_default)

        # Now, get location of cone vertex angles (all the rows in the 0th column)
        # cone_vertex_indices: np.ndarray = np.loadtxt(
        #     directory_temp / "temp-cones.txt", dtype=int)[:, 0]

        # NOTE: it seems that we may not need this?
        # At least testing with the bob duck mesh, using 2pi for the vertices worked just fine.
        # And it seems like a single island for the UV unwrapping is preferred to work fine.
        # Then, save the location of the cones into vertex_angles per Capouellez et al. 2023
        # vertex_angles[cone_vertex_indices] = np.pi    # * 3.0  # * random.random()

        # Finally, save to file...
        temp_file: pathlib.Path = pathlib.Path(directory_temp, "temp_Th_hat")
        np.savetxt(fname=temp_file, X=vertex_angles, newline="\n")

        # # TODO: move this functionality over to uv unwrap where it's more closely related.
        # # Now, write the angle file for this mesh, defaulting at 2pi
        # with open(temp_file, "w", encoding="utf8") as file:
        #     for _ in range(len(selected_object.data.vertices)):
        #         file.write(f"{(math.pi * 2.0)}\n")
        #         # file.write(f"{(math.pi * 1.0)}\n")

    def process_properties(self, properties: PropertyGroup) -> list[str]:
        """ Process list of properties and their values for input into arguments list.

        Args:
            properties (PropertyGroup): _description_

        Returns:
            list[str]: _description_
        """

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
