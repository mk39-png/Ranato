# pyright: reportInvalidTypeForm = false

import bpy.props
import bpy.types

# TODO: find a way to combine settings with the strategy via strong coupling...


class CampenSettings(bpy.types.PropertyGroup):
    """
    Concrete class holding properties associated with Campen et al. 2021's script arguments. Stores said arguments inside the .blend file itself in Blender's RNA data system.

    Which is to say that this class is exposed to/accessed by UVUnwrapperSettings()

    NOTE: attributes (also called properties if referring to Blender's version of the same data) must be named with the "prop_" prefix as many of the draw() and execute() scripts associated with UVUnwrapStrategy utilize getattr() and dir() for retrieving property values from settings.
    NOTE: attributes must also match the argument names in Campen et al 2021 script

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


class BFFSettings(bpy.types.PropertyGroup):
    """
    Settings for BFF conformal mapping method (if there are any settings in the future)
    """


class CEPSSettings(bpy.types.PropertyGroup):
    """
    Settings for CEPS algorithm
    """
    # curvatures file
    # scale factors file
    # TODO: may be worth implementing MPZ as another cone_mesh algorithm...
    # For identifying cones and whatnot... making that modular and expandable
    # ffield... from MPZ style crossfield

    # NOTE: descriptions from original repo
    # https://github.com/MarkGillespie/CEPS
    prop_greedyConeMaxU: bpy.props.IntProperty(
        name="Greedy Cones Maximum",
        description="Maximum allowed log scale factor when placing cones (lower value = lower distortion in final parameterization, default value=5)",
        default=5,
        min=1
    )
    prop_exactCones: bpy.props.BoolProperty(
        name="Use Exact Cones",
        description="Do not lump together nearby cones in the ffield input, if any. Cones prescribed via --curvatures or --scaleFactors are never lumped",
        default=False
    )
    prop_noFreeBoundary: bpy.props.BoolProperty(
        name="No Free Boundary",
        description="Do not impose minimal-area-distortion boundary conditions (useful, e.g. if prescribing polygonal boundary conditions i.e. specifying angles at vertices)",
        default=False
    )
    prop_viz: bpy.props.BoolProperty(
        # TODO: fix name making it more clear what this is doing
        name="Enable Polyscope GUI (external popup)",
        description="",
        default=False
    )


class CETMSettings(bpy.types.PropertyGroup):
    """
    Settings for CETM conformal mapping method (if there are any settings in the future)
    """


class UVUnwrapperSelection(bpy.types.PropertyGroup):
    """ Blender-facing and registered PropertyGroup that holds selected UV unwrapping algorithm via "method" attribute.

    Args:
        bpy (_type_): _description_
    """
    # NOTE: method for use in detecting which settings to display in the UI panel
    method: bpy.props.EnumProperty(
        name="UV Unwrapper",
        description="Choose preferred UV unwrapping algorithm",
        # NOTE: first element in each tuple must match bl_idname in STRATEGIES
        items=[
            ("CAMPEN", "Campen et al. 2021",
             "Efficient and Robust Discrete Conformal Equivalence with Boundary"),
            ("CEPS", "CEPS", "Discrete Conformal Equivalence of Polyhedral Surfaces"),
            ("CETM", "CETM", "Conformal Equivalence of Triangle Meshes"),
            ("BFF", "BFF", "Boundary First Flattening"),
        ],
        default="CAMPEN"
    )

    # Define properties to click on (and store inside .blend)
    campen: bpy.props.PointerProperty(type=CampenSettings)
    ceps: bpy.props.PointerProperty(type=CEPSSettings)
    cetm: bpy.props.PointerProperty(type=CETMSettings)
    bff: bpy.props.PointerProperty(type=BFFSettings)
