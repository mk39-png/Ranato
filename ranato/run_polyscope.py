"""
Simple script that is called via submodule to run Polyscope
"""

import polyscope as ps
import igl
import pathlib
import argparse
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Script to run Polyscope in the background.")
    parser.add_argument("--file", type=str, help="filepath to obj file", required=True)
    parser.add_argument("--camera", type=str, help="filepath to camera matrix file", required=True)
    args: argparse.Namespace = parser.parse_args()

    # Read in .obj and camera matrix files
    obj_filepath = pathlib.Path(args.file)
    camera_filepath = pathlib.Path(args.camera)
    V, uv, N, F, FT, FN = igl.readOBJ(obj_filepath)
    opengl_camera_matrix: np.ndarray = np.loadtxt(camera_filepath, delimiter=",")

    # Setup Polyscope and display mesh from a given camera matrix
    ps.init()
    ps_mesh: ps.SurfaceMesh = ps.register_surface_mesh("my mesh", V, F)
    ps.set_up_dir("y_up")
    ps.set_front_dir("z_front")
    # ps.set_front_dir("neg_z_front")
    ps.set_camera_view_matrix(opengl_camera_matrix)

    ps.show()


if __name__ == "__main__":
    main()
