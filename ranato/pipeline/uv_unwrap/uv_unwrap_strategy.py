"""
Holds strategy pattern for UV unwrapping (aka mesh parametrization/conformal equivalence) algorithms
"""

import pathlib
from abc import ABC, abstractmethod
from typing import Any

import bpy
import numpy as np
from bpy.types import Context, Object, PointerProperty, PropertyGroup, UILayout, bpy_prop_collection

from ...common import ADDON_ID


class UVUnwrapStrategy(ABC):

    """
    Abstract class providing structure for any mesh parametrization (UV unwrapping) algorithm
    utilizing a strategy pattern.
    This is "hidden" from Blender as in it is not directly stored in a current .blend project file.

    Attributes:
        _id (str): used for accessing strategy attribute from UVUnwrapperSettings for displaying options from dropdown (e.g. campen, bff, ceps, cetm)
        DEPRECATED _key (dict[str,str]): used for storing Blender RNA property names and their matching script argument name
        bl_idname (str): Blender RNA name for this concrete instance of this class
        bl_label (str): for display in draw()
    """
    _id: str = ""
    # _key: dict[str, str] = {}
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
        # TODO: the vertex angles format may differ between CEPS and Campen...
        # In that Campen does not have a spot to specify vertex indices...

        # TODO: ASSERT THAT MESH HAS BEEN SELECTED
        # TODO: check if it also has attribute as well...
        if context.scene.target_mesh is None:
            raise BaseException("No mesh has been selected! Please select a mesh to process")

        print(type(context.scene.vertex_angles), context.scene.vertex_angles[0].index)
        print(type(context.scene.vertex_angles), context.scene.vertex_angles[0].angle)
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

    def _process_property(self, arg: str, val: bool | int | float) -> list[str]:
        """ Process list of properties and their values for input into arguments list.
        Works for Campen et al. 2021 and (todo) CEPS, both of which rely on C++-style argument handling.

        Args:
            arg (str): argument name we're processing
            val (bool | int | float): value of property to turn into argument for script
            # DEPRECATED properties (PropertyGroup): _description_

        Returns:
            list[str]: _description_
        """
        if type(val) is int:
            print("int")
            return [f"--{arg}", str(val)]
        elif type(val) is bool:
            print("bool")
            return [f"--{arg}"] if val is True else []
        elif type(val) is float:
            print("float")
            # TODO: may need to specify precision of the value... hence separate "if" for float case
            return [f"--{arg}", str(val)]

        raise BaseException(f"No matching datatype! {val} is not of type int, bool, or float.")

    def draw(self, layout: UILayout, settings: PointerProperty) -> None:
        """
        Concrete method for Blender panels drawing only relevant fields to the UV unwrapping algorithm.

        Args:
            layout (UILayout): UI layout referenced
            settings (PointerProperty): settings to display in UI. Is either CampenSettings, CEPSSettings, BFFSettings, or CETMSettings
        """

        # print("LOOK, ", self._id)
        # print("LOOK HERE", settings)
        # print(settings.campen.prop_do_reduction)
        # print(type(settings.campen.prop_do_reduction))
        # print(self.process_properties("do_reduction", settings.campen.prop_do_reduction))
        # self.process_properties(settings)
        # Now, rather than using the list comprehension, could instead utilize a KEY with their
        # associated Blender RNA name and their script argument name, which is a lot better to do

        # Gets the currently selected option to display
        # e.g. cetm, campen, ceps, bff
        uv_setting: PointerProperty = getattr(settings, self._id)
        # print("Selected UV setting, ", uv_setting)

        # HACK: Forced to name attributes with prefix "prop" to filter out Blender properties from Pythonm attributes
        # TODO: instead, we can grab settings.KEYS if that works
        # NOTE: this might go wrong if for some reason there is something in the RNA
        properties: list[str] = [
            name for name in dir(uv_setting) if name.startswith("prop")
        ]
        # print("Properties: ", properties)

        # args = []
        for property_name in properties:
            # print("Name-value pair", getattr(uv_setting, name))
            # arg = self.process_properties(name, getattr(uv_setting, name))
            # args.extend(arg)
            layout.prop(uv_setting, property_name)
        # print(args)

    @abstractmethod
    def execute(self, context: Context, settings: PointerProperty) -> None:
        """
        Method for Blender operations executing actual UV unwrapping.

        Args:
            context (Context): Blender context variables to utilize
            settings (PointerProperty): settings to utilize for UV unwrapping execution
        """
        raise NotImplementedError
