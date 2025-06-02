import bpy
import array
import gpu

# Utilized for the shader.
from random import random
from mathutils import Vector
from gpu_extras.batch import batch_for_shader
from gpu_extras.presets import draw_texture_2d
import numpy as np

# https://docs.blender.org/api/current/bpy.types.RenderEngine.html


class RanatoRenderEngine(bpy.types.RenderEngine):
    # These three members are used by blender to set up the
    # RenderEngine; define its internal name, visible name and capabilities.
    bl_idname = "RANATO"
    bl_label = "Ranato"
    bl_use_preview = True

    # Init is called whenever a new render engine instance is created. Multiple
    # instances may exist at the same time, for example for a viewport and final
    # render.
    # Note the generic arguments signature, and the call to the parent class
    # `__init__` methods, which are required for Blender to create the underlying
    # `RenderEngine` data.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scene_data = None
        self.draw_data = None
        self.bl_use_eevee_viewport = True  # used for the "Materials" view of Blender
        # self.bl_use_gpu_context = True # what does this do?
        # TODO: within the init, it seems like I should also setup the Space3dView things

    # TODO: there's an error here with how 'super' object has no attribute '__del__'
    # When the render engine instance is destroy, this is called. Clean up any
    # render engine data here, for example stopping running render threads.
    # def __del__(self):
    #     super().__del__()

    # This is the method called by Blender for both final renders (F12) and
    # small preview for materials, world and lights.

    def render(self, depsgraph):
        scene = depsgraph.scene
        scale = scene.render.resolution_percentage / 100.0
        self.size_x = int(scene.render.resolution_x * scale)
        self.size_y = int(scene.render.resolution_y * scale)

        #
        # How to put stuff onto the frame buffer...
        # I know how it is for the GPU, but like, what about here???
        # TODO: use the viewport render for the render here... except, here we're using the
        #       primary camera of the scene
        #

        # Fill the render result with a flat color. The frame-buffer is
        # defined as a list of pixels, each pixel itself being a list of
        # R,G,B,A values.
        if self.is_preview:
            color = [0.1, 0.2, 0.1, 1.0]
        else:
            color = [0.2, 0.1, 0.1, 1.0]

        print("Rendering...")

        # Well, we need the material colors....
        # For each mesh in the scene, utilize its materials color...
        # Try starting with that...
        # Rather than this base color of stuff...

        # Rather than making this... utilize the Shader to write down some of the results...
        pixel_count = self.size_x * self.size_y
        rect = [color] * pixel_count

        # Here we write the pixel values to the RenderResult
        result = self.begin_result(0, 0, self.size_x, self.size_y)
        layer = result.layers[0].passes["Combined"]
        layer.rect = rect
        self.end_result(result)

    # For viewport renders, this method gets called once at the start and
    # whenever the scene or 3D viewport changes. This method is where data
    # should be read from Blender in the same thread. Typically a render
    # thread will be started to do the work while keeping Blender responsive.
    def view_update(self, context, depsgraph):
        region = context.region
        view3d = context.space_data
        scene = depsgraph.scene

        # --------------------------------------------
        #
        # print(depsgraph.objects[0])

        # OK, now with <bpy_struct, Object("Cube") at 0x000001F9EA42D608, evaluated>
        # We can now use Object
        # sample_mesh = depsgraph.objects[0].to_mesh(
        #     preserve_all_data_layers=True, depsgraph=depsgraph)

        # Now, mesh is returned!
        # What can we do with mesh???
        # print(sample_mesh.edges[0])

        #
        # --------------------------------------------

        # Get viewport dimensions
        dimensions = region.width, region.height

        if not self.scene_data:
            # First time initialization
            self.scene_data = []
            first_time = True

            # Loop over all datablocks used in the scene.
            for datablock in depsgraph.ids:
                pass
        else:
            first_time = False

            # Test which datablocks changed
            for update in depsgraph.updates:
                print("Datablock updated: ", update.id.name)

            # Test if any material was added, removed or changed.
            if depsgraph.id_type_updated('MATERIAL'):
                print("Materials updated")

        # Loop over all object instances in the scene.
        if first_time or depsgraph.id_type_updated('OBJECT'):
            for instance in depsgraph.object_instances:
                pass

    # For viewport renders, this method is called whenever Blender redraws
    # the 3D viewport. The renderer is expected to quickly draw the render
    # with OpenGL, and not perform other expensive work.
    # Blender will draw overlays for selection and editing on top of the
    # rendered image automatically.
    def view_draw(self, context, depsgraph):
        # Lazily import GPU module, so that the render engine works in
        # background mode where the GPU module can't be imported by default.
        region = context.region
        scene = depsgraph.scene

        # NOTE: below is from depsgraph documentation
        # for object_instance in depsgraph.object_instances:
        #     # This is an object which is being instanced.
        #     obj = object_instance.object
        #     # `is_instance` denotes whether the object is coming from instances (as an opposite of
        #     # being an emitting object. )
        #     if not object_instance.is_instance:
        #         print(f"Object {obj.name} at {object_instance.matrix_world}")
        #     else:
        #         # Instanced will additionally have fields like uv, random_id and others which are
        #         # specific for instances. See Python API for DepsgraphObjectInstance for details,
        #         print(
        #             f"Instance of {obj.name} at {object_instance.matrix_world}")

        # Get viewport dimensions
        dimensions = region.width, region.height

        # Bind shader that converts from scene linear to display space,
        gpu.state.blend_set('ALPHA_PREMULT')

        # TODO: Why is the binding of the space shader needed?
        # Binds GLSL fragment shader... to the scene using scene color management properties.
        self.bind_display_space_shader(scene)

        # TODO: this is only called once... when draw_data doesn't exist...
        if not self.draw_data or self.draw_data.dimensions != dimensions:
            self.draw_data = CustomDrawData(dimensions, depsgraph)

        # TODO: now, blender is supposed to draw its stuff on top...
        # But whenever we select any objects... Blender crashes.
        self.draw_data.draw()

        self.unbind_display_space_shader()
        gpu.state.blend_set('NONE')


# ------------------------------------------------------------------------------
#
# DRAWING DATA
#
# ------------------------------------------------------------------------------
# CustomDrawData used by view_draw() to make the viewport render!
# TODO: also, make sure that the viewport render is also what we get when using 'F12' render.
# Basically, think of this like the OpenGL side of the engine.
# While, the RanatoRenderEngine is the Blender side of the engine.

# TODO: put this inside a different file???
# TODO: MOVE THIS SOMEWHERE ELSE....
# This whole "custom draw data" messes everything up and makes life a lot miserable for us.
class CustomDrawData:
    # Need the depsgraph for scene meshes to render.
    def __init__(self, dimensions, depsgraph):
        # Generate dummy float image buffer
        self.dimensions = dimensions
        width, height = dimensions

        # TODO: below is gonna be a problem when we update the depsgraph and the below doesn't end up updating...
        # Basically, find a way to update this depsgraph from RanatoRenderEngine class
        self.depsgraph = depsgraph
        # -----------------------------------------------------------------------
        #
        # # Code to look through everything that's part of the scene...
        #
        #
        # print(depsgraph)
        # # https://blender.stackexchange.com/questions/33272/mesh-coordinates
        # sample_mesh = depsgraph.objects[0].to_mesh(
        #     preserve_all_data_layers=True, depsgraph=depsgraph)

        me = bpy.context.object.data

        # for poly in me.polygons:
        #     print("Polygon index: {:d}, length: {:d}".format(
        #         poly.index, poly.loop_total))

        #     # range is used here to show how the polygons reference loops,
        #     # for convenience 'poly.loop_indices' can be used instead.
        #     for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
        #         print("    Vertex: {:d}".format(
        #             me.loops[loop_index].vertex_index))

        # # So, grabbing the mesh... in local space or what?
        # for vert in sample_mesh.vertices:
        #     print(vert.co)
        # self.test_coords = sample_mesh.vertices

        # NOTE: these are the default colors, similar to how in CS351 I assigned default colors to the color buffer before placing in
        #       my actual values for said color.
        pixels = width * height * array.array('f', [0.024, 0.922, 0.89, 1.0])
        pixels = gpu.types.Buffer('FLOAT', width * height * 4, pixels)

        # Generate texture for render frame.
        self.texture = gpu.types.GPUTexture(
            (width, height), format='RGBA16F', data=pixels)

        # Note: This is just a didactic example.
        # In this case it would be more convenient to fill the texture with:
        # self.texture.clear('FLOAT', value=[0.1, 0.2, 0.1, 1.0])

        #
        #
        # TODO: does this support multiple meshes?
        # TODO: what is this with the batch shader and whatnot? And will this be rendered ON TOP OF THE BASE image???
        # mesh = depsgraph.objects[0]
        # mesh = mesh.find("meshes")
        # mesh = depsgraph.objects.get("meshes")

        # mesh = depsgraph.objects.find("meshes")
        mesh = me

        # if mesh != -1:
        mesh.calc_loop_triangles()

        vertices = np.empty((len(mesh.vertices), 3), 'f')
        indices = np.empty((len(mesh.loop_triangles), 3), 'i')

        mesh.vertices.foreach_get(
            "co", np.reshape(vertices, len(mesh.vertices) * 3))
        mesh.loop_triangles.foreach_get(
            "vertices", np.reshape(indices, len(mesh.loop_triangles) * 3))

        vertex_colors = [(random(), random(), random(), 1)
                         for _ in range(len(mesh.vertices))]

        self.shader = gpu.shader.from_builtin('SMOOTH_COLOR')
        # self.shader.uniform_float("color", (0, 0.5, 0.5, 1.0))

        # # OK, we got this batch... but where?
        # Apparently, this is recommended since it ensures that all vertex attributes necessary for a
        # specific shader are provided.
        # Batches should autmatically be drawn onto the Back Buffer, which then becomes the Front Buffer when all drawing is done
        self.batch = batch_for_shader(
            self.shader, 'TRIS',
            {"pos": vertices, "color": vertex_colors},
            indices=indices,
        )

        # TODO: instead of having separate draw method for this class... could just call draw_handler_add....
        # Then, when delete is called, remove draw_handler!
        # TODO: The mesh doesn't get update... which is unfortunate!
        bpy.types.SpaceView3D.draw_handler_add(
            self.draw, (), 'WINDOW', 'POST_VIEW')

        # self.shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        # self.batch = batch_for_shader(
        #     self.shader, 'TRIS', {"pos": vertices}, indices=indices)
        #
        #
        #

    def __del__(self):
        del self.texture

    # NOTE: this draw is called for the viewport render
    # TODO: for here, we want to utilize the SpaceView3d, actually???
    # Because that has a lot more things we want that are low level than here....
    # Though, could try using this...
    def draw(self):
        # Because I can't utilize Blender's offscreen texture for storing the frame buffer,
        #   I'll have to store it in the texture specified in __init__()
        # https://blender.stackexchange.com/questions/190140/copy-framebuffer-of-3d-view-into-custom-frame-buffer
        # get currently bound framebuffer
        # self.framebuffer = gpu.state.active_framebuffer_get()
        # self.framebuffer.clear(color=(0.0, 0.0, 0.0, 0.0))

        # So, below is supposed to do the acutal rendering onto the screen.
        # Problem is it drawing everything to be a certain color...
        # we just want to color the vertices...

        # print(gpu.state.active_framebuffer_get())

        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)
        # self.shader.bind()

        # Draws to the current framebuffer...
        self.batch.draw(self.shader)
        gpu.state.depth_mask_set(False)

        # draw_texture_2d(self.texture, (0, 0),
        #             self.texture.width, self.texture.height)

        # pixels = gpu.types.Buffer('FLOAT', 512 * 256 * 4, pixels)
        # width, height = self.dimensions

        # pixels = self.framebuffer.read_color(
        #     0, 0, width, height, 1, 0, 'UBYTE')

        # pixels2 = width * height * array.array('f', [0.024, 0.922, 0.89, 1.0])

        # pixels_buf = gpu.types.Buffer('FLOAT', width * height * 4, pixels2)

        # # Generate texture for render frame.
        # # self.texture = gpu.types.GPUTexture(
        # #     (2000, 2000), format='RGBA16F', data=pixels)

        # self.texture = gpu.types.GPUTexture(
        #     (width, height), format='RGBA16F', data=pixels_buf)

        # draw_texture_2d(self.texture, (0, 0),
        #                 self.texture.width, self.texture.height)


# ------------------------------------------------------------------------------
#
# BLENDER LOGISTICS SETUP
#
# ------------------------------------------------------------------------------
# RenderEngines also need to tell UI Panels that they are compatible with.
# We recommend to enable all panels marked as BLENDER_RENDER, and then
# exclude any panels that are replaced by custom panels registered by the
# render engine, or that are not supported.
def get_panels():
    exclude_panels = {
        'VIEWLAYER_PT_filter',
        'VIEWLAYER_PT_layer_passes',
    }

    panels = []

    for panel in bpy.types.Panel.__subclasses__():
        if hasattr(panel, 'COMPAT_ENGINES') and 'BLENDER_RENDER' in panel.COMPAT_ENGINES:
            if panel.__name__ not in exclude_panels:
                panels.append(panel)

    return panels


def register():
    # Register the RenderEngine
    bpy.utils.register_class(RanatoRenderEngine)

    # handle = bpy.types.SpaceView3D.draw_handler_add(
    #     RanatoRenderEngine.draw, (), 'WINDOW', 'POST_VIEW')

    for panel in get_panels():
        panel.COMPAT_ENGINES.add('RANATO')


def unregister():
    bpy.utils.unregister_class(RanatoRenderEngine)

    for panel in get_panels():
        if 'RANATO' in panel.COMPAT_ENGINES:
            panel.COMPAT_ENGINES.remove('RANATO')


if __name__ == "__main__":
    register()
