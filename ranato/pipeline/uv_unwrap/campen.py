# Setting up addon with multiple files
# https://blender.stackexchange.com/questions/202570/multi-files-to-addon

# Basic panel setup
# https://docs.blender.org/api/current/bpy.types.Panel.html

# Referencing below to see how Render Engine interacts with Panel
# (and changing it all so that the panel only appears when user activates Ranato)
# https://github.com/bnpr/Malt/blob/725f509ab25be736cb592cf1e9d5258ed4271e8a/BlenderMalt/MaltMaterial.py#L98
# https://github.com/Griperis/BlenderDataVis/blob/master/data_vis/operators/surface_chart.py


import subprocess
from typing import Any

import bpy.props
import bpy.types

from ...common import ADDON_ID, INPUT_OBJ_FILENAME
from .uv_unwrap_strategy import UVUnwrapStrategy
from ..vertex_angles import retrieveVertexAngles


# TODO: find a way to combine settings with the strategy via strong coupling...
class CampenSettings(bpy.types.PropertyGroup):
    """ 
    Concrete class holding properties associated with Campen et al. 2021's script arguments. Stores said arguments inside the .blend file itself in Blender's RNA data system.

    Which is to say that this class is exposed to/accessed by UVUnwrapperSettings()

    NOTE: attributes (also called properties if referring to Blender's version of the same data) must be named with the "prop_" prefix as many of the draw() and execute() scripts associated with UVUnwrapStrategy utilize getattr() and dir() for retrieving property values from settings.

    Attributes:
        prop_use_mpf (BoolProperty): multiprecision enabling
        prop_do_reduction
        prop_prec
        prop_max_itr
        prop_energy_cond
        prop_no_round_Th_hat
        prop_no_lm_reset
        prop_eps
        prop_lambda0
        prop_bound_norm_thres
    """

    # NOTE: descriptions from original script arguments
    # https://github.com/mk39-png/ConformalIdealDelaunay/blob/master/py/script_conformal.py
    prop_use_mpf: bpy.props.BoolProperty(
        name="Use MPF",
        description="True for enable multiprecision",
        default=False
    )
    prop_do_reduction: bpy.props.BoolProperty(
        name="Do Reduction",
        description="Do reduction for search direction",
        default=False
    )
    prop_prec: bpy.props.IntProperty(
        name="Precision",
        description="Choose the mantissa value of MPF",
        default=10  # TODO: check the actual default precision...
    )
    prop_max_itr: bpy.props.IntProperty(
        name="Max Iterations",
        description="Choose the maximum number of iterations",
        default=50,
        min=0
    )

    prop_energy_cond: bpy.props.BoolProperty(
        name="Energy Condition",
        description="True for enable energy computation for line-search",
        default=False
    )
    prop_no_round_Th_hat: bpy.props.BoolProperty(
        name="No Round Th_hat",
        description="True for NOT rounding Th_hat (i.e. vertex angles) values to multiples of pi/60",
        default=False
    )

    prop_no_lm_reset: bpy.props.BoolProperty(
        name="No LM Reset",
        description="True for using double the previous lambda for line search.",
        default=False
    )

    prop_eps: bpy.props.FloatProperty(
        name="eps",
        description="Target error threshold",
        default=0.0,
    )

    prop_lambda0: bpy.props.FloatProperty(
        name="lambda0",
        description="Initial lambda value",
        default=1.0
    )

    prop_bound_norm_thres: bpy.props.FloatProperty(
        name="Bound Norm threshold",
        description="Threshold to drop the norm bound",
        default=1e-10
    )

# TODO: rename to CampenImplementation?


class CampenStrategy(UVUnwrapStrategy):
    _id = "campen"

    # TODO: why don't I just have function calls below? That way, it's a lot easier...
    # By that, I mean holt the reference to the attributes...
    # _key = {
    #     "prop_use_mpf": "use_mpf",
    #     "prop_do_reduction": "do_reduction",
    #     "prop_prec": "prec",
    #     "prop_max_itr": "max_itr",
    #     "prop_energy_cond": "energy_cond",
    #     "prop_no_round_Th_hat": "no_round_Th_hat",
    #     "prop_no_lm_reset": "no_lm_reset",
    #     "prop_eps": "eps",
    #     "prop_lambda0": "lambda0",
    #     "prop_bound_norm_thres": "bound_norm_thres",
    # }

    bl_idname = "CAMPEN"
    bl_label = "Campen et al. 2021"

    def _call_uv_unwrapper(self, args) -> Any:
        """
        """
        subprocess.run(args, check=False)

    # TODO: have something that converts bool to arguments and whatnot...

    def execute(self, context, settings):
        """
        # TODO: refactor so that function is universal for any UV unwrapper
        # TODO: this means passing in the args and params and whatnot...

        Calls the uv_unwrapper algorithm with specified commands.
        Also calls to activate the Conda environment if that has not been done already.
        Args:
            context (_type_): _description_
            settings (_type_): _description_
        """
        # Call sys subprocess to execute python script in another file.
        filepath_conda: str = bpy.context.preferences.addons[
            ADDON_ID].preferences.filepath_conda
        file_path_uv_unwrapper: str = bpy.context.preferences.addons[
            ADDON_ID].preferences.filepath_uv_unwrap_campen
        directory_temp: str = bpy.context.preferences.addons[
            ADDON_ID].preferences.directory_temp

        # TODO: retrieve values from Vertex Angles section...
        # TODO: have a check to make sure that file_path_uv_unwrapper preference has been checked.
        # TODO: is there a better way of doing this?
        # print(directory_temp)
        # print(dir(settings))
        properties: list[str] = [
            name for name in dir(settings.campen) if name.startswith("prop")
        ]

        # Now with the name of our args (converted from Blender property name to script argument names), we can find their value to pass into the script
        user_args: list[str] = []
        for property in properties:
            arg = property.replace("prop_", "")
            user_arg = self._process_property(arg, getattr(settings.campen, property))
            user_args.extend(user_arg)

        print("properties:", properties)
        print("user_args:", user_args)

        # TODO: fix below method to ONLY grab numpy array of indices and their values
        retrieveVertexAngles(context.scene.vertex_angles)
        self._retrieve_vertex_angles(context)

        # TODO: have a way to save the parameters/settings, especially with a mesh and its user-inputted cone vertices.
        # TODO: also for CEPS, need a way to track its parameters as well

        # TODO: allow user to specify parameters into this... by writing down whatever in the panel box
        script_args = [
            filepath_conda,
            file_path_uv_unwrapper,
            "--input", directory_temp,
            "--fname", INPUT_OBJ_FILENAME,
            "--output", directory_temp,
            "--output_type", "param",
            "--output_format", "obj",
            "--error_log"
        ]
        script_args.extend(user_args)
        print(script_args)
        self._call_uv_unwrapper(script_args)
        # TODO: perform verification on whether OBJ output is VALID!!!
