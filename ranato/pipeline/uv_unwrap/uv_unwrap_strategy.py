"""
Holds strategy pattern for UV unwrapping (aka mesh parametrization/conformal equivalence) algorithms
"""

import pathlib
import subprocess
from abc import ABC, abstractmethod
from typing import Any

import bpy
import numpy as np
from bpy.types import Context, Object, UILayout

from ...common import ADDON_ID
from ..vertex_angles import retrieve_cone_vertex_angles
from .uv_unwrap_settings import (BFFSettings, CampenSettings, CEPSSettings,
                                 CETMSettings, UVUnwrapperSelection)


class UVUnwrapStrategy(ABC):
    """
    Abstract class providing structure for any mesh parametrization (UV unwrapping) algorithm
    utilizing a strategy pattern.
    This is "hidden" from Blender as in it is not directly stored in a current .blend project file.

    Attributes:
        _id (str): used for accessing strategy attribute from UVUnwrapperSettings for displaying options from dropdown (e.g. campen, bff, ceps, cetm)
        bl_idname (str): Blender RNA name for this concrete instance of this class
        bl_label (str): for display in draw()
    """
    _id: str = ""
    bl_idname: str = ""
    bl_label: str = ""

    def _call_uv_unwrapper(self, args: list[str]) -> Any:
        """
        Internal helper function that calls external script if needed.

        Args:
            args (list[str]): list of arguments to pass into the script or executable being called
        """
        subprocess.run(args, check=False)

        process: subprocess.Popen[str] = subprocess.Popen(args,
                                                          stdout=subprocess.PIPE,
                                                          text=True)

        # TODO: have process timer to occassionally ping this process!
        # As in, call the below function on a timer... every couple of seconds or so.
        #
        process.poll()

        # TODO: when the process is done, then

    def _process_single_property(self, arg: str, val: bool | int | float
                                 ) -> tuple[str, str] | tuple[str] | tuple[()]:
        """ Process list of properties and their values for input into arguments list.
        Works for Campen et al. 2021 and (todo) CEPS, both of which rely on C++-style argument handling.


        :param arg: argument name we're processing
        :type arg: str
        :param val: value of property to turn into argument for script
        :type val: (bool | int | float)

        :return: (--key, value) if int/float val or (--key) if boolean val
        :rtype: tuple[str, str] | tuple[str] | tuple[()] 
        """
        # NOTE: need strict type comparisons here rather than isinstance() since apparently "bool" is a subclass of "int" in Python
        if type(val) is int:
            return (f"--{arg}", str(val))
        elif type(val) is bool:
            return (f"--{arg}",) if val is True else tuple()
        elif type(val) is float:
            # TODO: may need to specify precision of the value... hence separate "if" for float case
            return (f"--{arg}", str(val))

        raise ValueError(f"No matching datatype! {val} is not of type int, bool, or float.")

    def _process_properties(self, settings: UVUnwrapperSelection, ceps_format=False) -> list[str]:
        """ Takes a UV unwrapping property group (i.e. settings.campen, settings.ceps) and converts
        values of its properties into script arguments

        :return list[str]: list of arguments and their accompanying values if applicable
        """
        uv_setting: CampenSettings | CEPSSettings | BFFSettings | CETMSettings = getattr(
            settings, self._id)

        properties: list[str] = [
            name for name in dir(uv_setting) if name.startswith("prop")
        ]

        user_args: list[str] = []

        for prop in properties:
            arg: str = prop.replace("prop_", "")
            value: float | bool | int = getattr(uv_setting, prop)
            user_arg: tuple = self._process_single_property(arg, value)

            # NOTE: this is since CEPS requires us to write out the "=" sign when specifying
            #       user input for numerical arguments
            if ceps_format and (
                    type(value) is int or type(value) is float):
                arg_key: str = user_arg[0]
                arg_val: str = user_arg[1]
                user_arg = (f"{arg_key}={arg_val}",)
            user_args.extend(user_arg)

        return user_args

    def draw(self, layout: UILayout, settings: UVUnwrapperSelection) -> None:
        """
        Concrete method for Blender panels drawing only relevant fields to the UV unwrapping algorithm.

        Args:
            layout (UILayout): UI layout referenced
            settings (PointerProperty): settings to display in UI. Is either CampenSettings, CEPSSettings, BFFSettings, or CETMSettings
        """
        # Gets the currently selected option to display
        # e.g. cetm, campen, ceps, bff
        uv_setting: CampenSettings | CEPSSettings | BFFSettings | CETMSettings = getattr(
            settings, self._id)

        # HACK: Forced to name attributes with prefix "prop" to filter out Blender properties from Python attributes
        # TODO: instead, we can grab settings.KEYS if that works
        # NOTE: this might go wrong if for some reason there is something in the RNA
        properties: list[str] = [
            name for name in dir(uv_setting) if name.startswith("prop")
        ]

        for property_name in properties:
            layout.prop(uv_setting, property_name)

    @abstractmethod
    def execute(self, context: Context, settings: UVUnwrapperSelection) -> None:
        """
        Method for Blender operations executing actual UV unwrapping.

        Args:
            context (Context): Blender context variables to utilize
            settings (PointerProperty): settings to utilize for UV unwrapping execution
        """
        raise NotImplementedError("Execute has not been implemented!")
