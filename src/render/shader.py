
# import bpy
# import gpu
# import math
# import random
# from mathutils import Matrix
# from gpu_extras.batch import batch_for_shader
# import gpu

# # create the 'UNIFORM COLOR' shader ourself

# shader_info = gpu.types.GPUShaderCreateInfo()
# shader_info.push_constant('MAT4', "viewProjectionMatrix")
# shader_info.push_constant('FLOAT', "color")
# shader_info.vertex_in(0, 'VEC3', "position")
# shader_info.fragment_out(0, 'VEC4', "FragColor")

# shader_info.vertex_source(
#     "void main()"
#     "{"
#     "  gl_Position = viewProjectionMatrix * vec4(position, 1.0f);"
#     "}"
# )

# shader_info.fragment_source(
#     "void main()"
#     "{"
#     "  float c = color / 255.0;"
#     "  FragColor = vec4(c, c, c, 1.0);"
#     "}"
# )

# shader = gpu.shader.create_from_info(shader_info)
# del shader_info

# # create the batches to render each column in the image

# batches = []
# step = 2 / 255.0
# for i in range(256):
#     k = i * step - 1.0 - step * 0.5
#     coords = [
#         (k, -1, 0), (k+step, -1, 0), (k+step, 1, 0),
#         (k+step, 1, 0), (k, 1, 0), (k, -1, 0)]

#     batches.append(batch_for_shader(shader, 'TRIS', {"position": coords}))

# # start offscreen rendering

# IMAGE_NAME = "all_hex_values"
# WIDTH = 256
# HEIGHT = 256

# offscreen = gpu.types.GPUOffScreen(WIDTH, HEIGHT)

# with offscreen.bind():
#     fb = gpu.state.active_framebuffer_get()
#     fb.clear(color=(0.0, 0.0, 0.0, 0.0))
#     with gpu.matrix.push_pop():
#         # reset matrices -> use normalized device coordinates [-1, 1]
#         shader.uniform_float("viewProjectionMatrix", Matrix.Identity(4))

#         for i in range(256):
#             shader.uniform_float("color", i)
#             batches[i].draw(shader)

#     buffer = fb.read_color(0, 0, WIDTH, HEIGHT, 4, 0, 'UBYTE')

# offscreen.free()

# # copy the render result into an blender image

# if IMAGE_NAME not in bpy.data.images:
#     bpy.data.images.new(IMAGE_NAME, WIDTH, HEIGHT)

# image = bpy.data.images[IMAGE_NAME]
# image.scale(WIDTH, HEIGHT)

# buffer.dimensions = WIDTH * HEIGHT * 4
# image.pixels = [v / 255 for v in buffer]


import bpy
import gpu

offscreen = gpu.types.GPUOffScreen(512, 256)


def draw_gpu(self):
    from gpu_extras.presets import draw_texture_2d

    # TODO: build up a 2d texture to render onto the screen...
    # Rendering a 3D scene into a texture! Yes! This was the step I was forgetting...
    # https://docs.blender.org/api/current/gpu.html#rendering-the-3d-view-into-a-texture
    context = bpy.context
    scene = context.scene
    width, height = self.dimensions
    view_matrix = scene.camera.matrix_world.inverted()

    projection_matrix = scene.camera.calc_matrix_camera(
        context.evaluated_depsgraph_get(), x=width, y=height)

    offscreen.draw_view3d(
        scene,
        context.view_layer,
        context.space_data,
        context.region,
        view_matrix,
        projection_matrix,
        do_color_management=True)

    gpu.state.depth_mask_set(False)
    draw_texture_2d(offscreen.texture_color,
                    (10, 10), width, height)
