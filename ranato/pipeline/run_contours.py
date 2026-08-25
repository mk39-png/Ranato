# TEST SCRIPT TO RUN CODE WITHOUT DEBUG SLOWDOWN

import pathlib

import numpy as np
from pyalgcon.pipelines.generate_algebraic_contours import \
    generate_algebraic_contours

cam_file = pathlib.Path(r"D:\Repos\Ranato\ranato\__temp__\temp_camera_matrix.csv")
obj = pathlib.Path(r"D:\Repos\Ranato\ranato\__temp__\temp_out.obj")

camera_matrix = np.loadtxt(cam_file, delimiter=",")
print(camera_matrix)

generate_algebraic_contours(camera_matrix, obj)
