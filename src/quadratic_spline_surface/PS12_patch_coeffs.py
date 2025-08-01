import numpy as np


def PS12_patch_coeffs() -> np.ndarray:
    """
    Returns hard-coded numpy array of shape (12, 6, 12)
    """
    patch_coeffs: np.ndarray = np.ndarray(shape=(12, 6, 12), dtype=float)

    patch_coeffs[0][0][0] = 0
    patch_coeffs[0][0][1] = 0
    patch_coeffs[0][0][2] = 1
    patch_coeffs[0][0][3] = 0.3e1 / 0.8e1
    patch_coeffs[0][0][4] = -0.3e1 / 0.8e1
    patch_coeffs[0][0][5] = 0.3e1 / 0.8e1
    patch_coeffs[0][0][6] = -0.3e1 / 0.8e1
    patch_coeffs[0][0][7] = 0.1e1 / 0.8e1
    patch_coeffs[0][0][8] = 0.1e1 / 0.8e1
    patch_coeffs[0][0][9] = 0.1e1 / 0.2e1
    patch_coeffs[0][0][10] = -0.1e1 / 0.2e1
    patch_coeffs[0][0][11] = -0.1e1 / 0.2e1
    patch_coeffs[0][1][0] = 0
    patch_coeffs[0][1][1] = 0
    patch_coeffs[0][1][2] = 0
    patch_coeffs[0][1][3] = -0.3e1 / 0.2e1
    patch_coeffs[0][1][4] = 0.3e1 / 0.2e1
    patch_coeffs[0][1][5] = -0.3e1 / 0.2e1
    patch_coeffs[0][1][6] = 0.3e1 / 0.2e1
    patch_coeffs[0][1][7] = 0.1e1 / 0.2e1
    patch_coeffs[0][1][8] = -0.1e1 / 0.2e1
    patch_coeffs[0][1][9] = -2
    patch_coeffs[0][1][10] = 2
    patch_coeffs[0][1][11] = 2
    patch_coeffs[0][2][0] = 0
    patch_coeffs[0][2][1] = 0
    patch_coeffs[0][2][2] = 0
    patch_coeffs[0][2][3] = -0.3e1 / 0.2e1
    patch_coeffs[0][2][4] = 0.3e1 / 0.2e1
    patch_coeffs[0][2][5] = -0.3e1 / 0.2e1
    patch_coeffs[0][2][6] = 0.3e1 / 0.2e1
    patch_coeffs[0][2][7] = -0.1e1 / 0.2e1
    patch_coeffs[0][2][8] = 0.1e1 / 0.2e1
    patch_coeffs[0][2][9] = -2
    patch_coeffs[0][2][10] = 2
    patch_coeffs[0][2][11] = 2
    patch_coeffs[0][3][0] = 0
    patch_coeffs[0][3][1] = 2
    patch_coeffs[0][3][2] = -2
    patch_coeffs[0][3][3] = 3
    patch_coeffs[0][3][4] = -3
    patch_coeffs[0][3][5] = 3
    patch_coeffs[0][3][6] = -0.5e1 / 0.2e1
    patch_coeffs[0][3][7] = -1
    patch_coeffs[0][3][8] = 0.1e1 / 0.2e1
    patch_coeffs[0][3][9] = 4
    patch_coeffs[0][3][10] = -2
    patch_coeffs[0][3][11] = -4
    patch_coeffs[0][4][0] = 3
    patch_coeffs[0][4][1] = -1
    patch_coeffs[0][4][2] = -2
    patch_coeffs[0][4][3] = 0.3e1 / 0.2e1
    patch_coeffs[0][4][4] = -0.3e1 / 0.4e1
    patch_coeffs[0][4][5] = 0.3e1 / 0.2e1
    patch_coeffs[0][4][6] = -0.7e1 / 0.4e1
    patch_coeffs[0][4][7] = -0.1e1 / 0.4e1
    patch_coeffs[0][4][8] = -0.1e1 / 0.4e1
    patch_coeffs[0][4][9] = 2
    patch_coeffs[0][4][10] = -3
    patch_coeffs[0][4][11] = -1
    patch_coeffs[0][5][0] = 0
    patch_coeffs[0][5][1] = 2
    patch_coeffs[0][5][2] = -2
    patch_coeffs[0][5][3] = 0.3e1 / 0.2e1
    patch_coeffs[0][5][4] = -0.3e1 / 0.2e1
    patch_coeffs[0][5][5] = 0.3e1 / 0.2e1
    patch_coeffs[0][5][6] = -1
    patch_coeffs[0][5][7] = 0.1e1 / 0.2e1
    patch_coeffs[0][5][8] = -1
    patch_coeffs[0][5][9] = 2
    patch_coeffs[0][5][10] = -2
    patch_coeffs[0][5][11] = -2
    patch_coeffs[1][0][0] = 0
    patch_coeffs[1][0][1] = 0
    patch_coeffs[1][0][2] = 1
    patch_coeffs[1][0][3] = 0.3e1 / 0.8e1
    patch_coeffs[1][0][4] = -0.3e1 / 0.8e1
    patch_coeffs[1][0][5] = 0.3e1 / 0.8e1
    patch_coeffs[1][0][6] = -0.3e1 / 0.8e1
    patch_coeffs[1][0][7] = 0.1e1 / 0.8e1
    patch_coeffs[1][0][8] = 0.1e1 / 0.8e1
    patch_coeffs[1][0][9] = 0.1e1 / 0.2e1
    patch_coeffs[1][0][10] = -0.1e1 / 0.2e1
    patch_coeffs[1][0][11] = -0.1e1 / 0.2e1
    patch_coeffs[1][1][0] = 0
    patch_coeffs[1][1][1] = 0
    patch_coeffs[1][1][2] = 0
    patch_coeffs[1][1][3] = -0.3e1 / 0.2e1
    patch_coeffs[1][1][4] = 0.3e1 / 0.2e1
    patch_coeffs[1][1][5] = -0.3e1 / 0.2e1
    patch_coeffs[1][1][6] = 0.3e1 / 0.2e1
    patch_coeffs[1][1][7] = 0.1e1 / 0.2e1
    patch_coeffs[1][1][8] = -0.1e1 / 0.2e1
    patch_coeffs[1][1][9] = -2
    patch_coeffs[1][1][10] = 2
    patch_coeffs[1][1][11] = 2
    patch_coeffs[1][2][0] = 0
    patch_coeffs[1][2][1] = 0
    patch_coeffs[1][2][2] = 0
    patch_coeffs[1][2][3] = -0.3e1 / 0.2e1
    patch_coeffs[1][2][4] = 0.3e1 / 0.2e1
    patch_coeffs[1][2][5] = -0.3e1 / 0.2e1
    patch_coeffs[1][2][6] = 0.3e1 / 0.2e1
    patch_coeffs[1][2][7] = -0.1e1 / 0.2e1
    patch_coeffs[1][2][8] = 0.1e1 / 0.2e1
    patch_coeffs[1][2][9] = -2
    patch_coeffs[1][2][10] = 2
    patch_coeffs[1][2][11] = 2
    patch_coeffs[1][3][0] = 2
    patch_coeffs[1][3][1] = 0
    patch_coeffs[1][3][2] = -2
    patch_coeffs[1][3][3] = 3
    patch_coeffs[1][3][4] = -0.5e1 / 0.2e1
    patch_coeffs[1][3][5] = 3
    patch_coeffs[1][3][6] = -3
    patch_coeffs[1][3][7] = 0.1e1 / 0.2e1
    patch_coeffs[1][3][8] = -1
    patch_coeffs[1][3][9] = 4
    patch_coeffs[1][3][10] = -4
    patch_coeffs[1][3][11] = -2
    patch_coeffs[1][4][0] = 2
    patch_coeffs[1][4][1] = 0
    patch_coeffs[1][4][2] = -2
    patch_coeffs[1][4][3] = 0.3e1 / 0.2e1
    patch_coeffs[1][4][4] = -1
    patch_coeffs[1][4][5] = 0.3e1 / 0.2e1
    patch_coeffs[1][4][6] = -0.3e1 / 0.2e1
    patch_coeffs[1][4][7] = -1
    patch_coeffs[1][4][8] = 0.1e1 / 0.2e1
    patch_coeffs[1][4][9] = 2
    patch_coeffs[1][4][10] = -2
    patch_coeffs[1][4][11] = -2
    patch_coeffs[1][5][0] = -1
    patch_coeffs[1][5][1] = 3
    patch_coeffs[1][5][2] = -2
    patch_coeffs[1][5][3] = 0.3e1 / 0.2e1
    patch_coeffs[1][5][4] = -0.7e1 / 0.4e1
    patch_coeffs[1][5][5] = 0.3e1 / 0.2e1
    patch_coeffs[1][5][6] = -0.3e1 / 0.4e1
    patch_coeffs[1][5][7] = -0.1e1 / 0.4e1
    patch_coeffs[1][5][8] = -0.1e1 / 0.4e1
    patch_coeffs[1][5][9] = 2
    patch_coeffs[1][5][10] = -1
    patch_coeffs[1][5][11] = -3
    patch_coeffs[2][0][0] = -1
    patch_coeffs[2][0][1] = 0
    patch_coeffs[2][0][2] = 2
    patch_coeffs[2][0][3] = 0.1e1 / 0.8e1
    patch_coeffs[2][0][4] = -0.3e1 / 0.8e1
    patch_coeffs[2][0][5] = -0.3e1 / 0.8e1
    patch_coeffs[2][0][6] = 0.3e1 / 0.8e1
    patch_coeffs[2][0][7] = 0.1e1 / 0.8e1
    patch_coeffs[2][0][8] = 0.3e1 / 0.8e1
    patch_coeffs[2][0][9] = -0.1e1 / 0.2e1
    patch_coeffs[2][0][10] = 0.1e1 / 0.2e1
    patch_coeffs[2][0][11] = -0.1e1 / 0.2e1
    patch_coeffs[2][1][0] = 4
    patch_coeffs[2][1][1] = 0
    patch_coeffs[2][1][2] = -4
    patch_coeffs[2][1][3] = -0.1e1 / 0.2e1
    patch_coeffs[2][1][4] = 0.3e1 / 0.2e1
    patch_coeffs[2][1][5] = 0.3e1 / 0.2e1
    patch_coeffs[2][1][6] = -0.3e1 / 0.2e1
    patch_coeffs[2][1][7] = 0.1e1 / 0.2e1
    patch_coeffs[2][1][8] = -0.3e1 / 0.2e1
    patch_coeffs[2][1][9] = 2
    patch_coeffs[2][1][10] = -2
    patch_coeffs[2][1][11] = 2
    patch_coeffs[2][2][0] = 2
    patch_coeffs[2][2][1] = 0
    patch_coeffs[2][2][2] = -2
    patch_coeffs[2][2][3] = -1
    patch_coeffs[2][2][4] = 0.3e1 / 0.2e1
    patch_coeffs[2][2][5] = 0
    patch_coeffs[2][2][6] = 0
    patch_coeffs[2][2][7] = -0.1e1 / 0.2e1
    patch_coeffs[2][2][8] = 0
    patch_coeffs[2][2][9] = 0
    patch_coeffs[2][2][10] = 0
    patch_coeffs[2][2][11] = 2
    patch_coeffs[2][3][0] = -2
    patch_coeffs[2][3][1] = 0
    patch_coeffs[2][3][2] = 2
    patch_coeffs[2][3][3] = 2
    patch_coeffs[2][3][4] = -0.5e1 / 0.2e1
    patch_coeffs[2][3][5] = 0
    patch_coeffs[2][3][6] = 0
    patch_coeffs[2][3][7] = 0.1e1 / 0.2e1
    patch_coeffs[2][3][8] = 0
    patch_coeffs[2][3][9] = 0
    patch_coeffs[2][3][10] = 0
    patch_coeffs[2][3][11] = -2
    patch_coeffs[2][4][0] = -2
    patch_coeffs[2][4][1] = 0
    patch_coeffs[2][4][2] = 2
    patch_coeffs[2][4][3] = 0.1e1 / 0.2e1
    patch_coeffs[2][4][4] = -1
    patch_coeffs[2][4][5] = -0.3e1 / 0.2e1
    patch_coeffs[2][4][6] = 0.3e1 / 0.2e1
    patch_coeffs[2][4][7] = -1
    patch_coeffs[2][4][8] = 0.3e1 / 0.2e1
    patch_coeffs[2][4][9] = -2
    patch_coeffs[2][4][10] = 2
    patch_coeffs[2][4][11] = -2
    patch_coeffs[2][5][0] = -2
    patch_coeffs[2][5][1] = 3
    patch_coeffs[2][5][2] = -1
    patch_coeffs[2][5][3] = 0.5e1 / 0.4e1
    patch_coeffs[2][5][4] = -0.7e1 / 0.4e1
    patch_coeffs[2][5][5] = 0.3e1 / 0.4e1
    patch_coeffs[2][5][6] = 0
    patch_coeffs[2][5][7] = -0.1e1 / 0.4e1
    patch_coeffs[2][5][8] = 0
    patch_coeffs[2][5][9] = 1
    patch_coeffs[2][5][10] = 0
    patch_coeffs[2][5][11] = -3
    patch_coeffs[3][0][0] = -1
    patch_coeffs[3][0][1] = -1
    patch_coeffs[3][0][2] = 3
    patch_coeffs[3][0][3] = -0.5e1 / 0.8e1
    patch_coeffs[3][0][4] = 0.3e1 / 0.8e1
    patch_coeffs[3][0][5] = -0.5e1 / 0.8e1
    patch_coeffs[3][0][6] = 0.3e1 / 0.8e1
    patch_coeffs[3][0][7] = 0.3e1 / 0.8e1
    patch_coeffs[3][0][8] = 0.3e1 / 0.8e1
    patch_coeffs[3][0][9] = -0.3e1 / 0.2e1
    patch_coeffs[3][0][10] = 0.1e1 / 0.2e1
    patch_coeffs[3][0][11] = 0.1e1 / 0.2e1
    patch_coeffs[3][1][0] = 4
    patch_coeffs[3][1][1] = 2
    patch_coeffs[3][1][2] = -6
    patch_coeffs[3][1][3] = 1
    patch_coeffs[3][1][4] = 0
    patch_coeffs[3][1][5] = 2
    patch_coeffs[3][1][6] = -0.3e1 / 0.2e1
    patch_coeffs[3][1][7] = 0
    patch_coeffs[3][1][8] = -0.3e1 / 0.2e1
    patch_coeffs[3][1][9] = 4
    patch_coeffs[3][1][10] = -2
    patch_coeffs[3][1][11] = 0
    patch_coeffs[3][2][0] = 2
    patch_coeffs[3][2][1] = 4
    patch_coeffs[3][2][2] = -6
    patch_coeffs[3][2][3] = 2
    patch_coeffs[3][2][4] = -0.3e1 / 0.2e1
    patch_coeffs[3][2][5] = 1
    patch_coeffs[3][2][6] = 0
    patch_coeffs[3][2][7] = -0.3e1 / 0.2e1
    patch_coeffs[3][2][8] = 0
    patch_coeffs[3][2][9] = 4
    patch_coeffs[3][2][10] = 0
    patch_coeffs[3][2][11] = -2
    patch_coeffs[3][3][0] = -2
    patch_coeffs[3][3][1] = -4
    patch_coeffs[3][3][2] = 6
    patch_coeffs[3][3][3] = -1
    patch_coeffs[3][3][4] = 0.1e1 / 0.2e1
    patch_coeffs[3][3][5] = -1
    patch_coeffs[3][3][6] = 0
    patch_coeffs[3][3][7] = 0.3e1 / 0.2e1
    patch_coeffs[3][3][8] = 0
    patch_coeffs[3][3][9] = -4
    patch_coeffs[3][3][10] = 0
    patch_coeffs[3][3][11] = 2
    patch_coeffs[3][4][0] = -2
    patch_coeffs[3][4][1] = -1
    patch_coeffs[3][4][2] = 3
    patch_coeffs[3][4][3] = -0.1e1 / 0.4e1
    patch_coeffs[3][4][4] = -0.1e1 / 0.4e1
    patch_coeffs[3][4][5] = -0.7e1 / 0.4e1
    patch_coeffs[3][4][6] = 0.3e1 / 0.2e1
    patch_coeffs[3][4][7] = -0.3e1 / 0.4e1
    patch_coeffs[3][4][8] = 0.3e1 / 0.2e1
    patch_coeffs[3][4][9] = -3
    patch_coeffs[3][4][10] = 2
    patch_coeffs[3][4][11] = -1
    patch_coeffs[3][5][0] = -2
    patch_coeffs[3][5][1] = -1
    patch_coeffs[3][5][2] = 3
    patch_coeffs[3][5][3] = -0.7e1 / 0.4e1
    patch_coeffs[3][5][4] = 0.5e1 / 0.4e1
    patch_coeffs[3][5][5] = -0.1e1 / 0.4e1
    patch_coeffs[3][5][6] = 0
    patch_coeffs[3][5][7] = 0.3e1 / 0.4e1
    patch_coeffs[3][5][8] = 0
    patch_coeffs[3][5][9] = -3
    patch_coeffs[3][5][10] = 0
    patch_coeffs[3][5][11] = 1
    patch_coeffs[4][0][0] = -1
    patch_coeffs[4][0][1] = -1
    patch_coeffs[4][0][2] = 3
    patch_coeffs[4][0][3] = -0.5e1 / 0.8e1
    patch_coeffs[4][0][4] = 0.3e1 / 0.8e1
    patch_coeffs[4][0][5] = -0.5e1 / 0.8e1
    patch_coeffs[4][0][6] = 0.3e1 / 0.8e1
    patch_coeffs[4][0][7] = 0.3e1 / 0.8e1
    patch_coeffs[4][0][8] = 0.3e1 / 0.8e1
    patch_coeffs[4][0][9] = -0.3e1 / 0.2e1
    patch_coeffs[4][0][10] = 0.1e1 / 0.2e1
    patch_coeffs[4][0][11] = 0.1e1 / 0.2e1
    patch_coeffs[4][1][0] = 4
    patch_coeffs[4][1][1] = 2
    patch_coeffs[4][1][2] = -6
    patch_coeffs[4][1][3] = 1
    patch_coeffs[4][1][4] = 0
    patch_coeffs[4][1][5] = 2
    patch_coeffs[4][1][6] = -0.3e1 / 0.2e1
    patch_coeffs[4][1][7] = 0
    patch_coeffs[4][1][8] = -0.3e1 / 0.2e1
    patch_coeffs[4][1][9] = 4
    patch_coeffs[4][1][10] = -2
    patch_coeffs[4][1][11] = 0
    patch_coeffs[4][2][0] = 2
    patch_coeffs[4][2][1] = 4
    patch_coeffs[4][2][2] = -6
    patch_coeffs[4][2][3] = 2
    patch_coeffs[4][2][4] = -0.3e1 / 0.2e1
    patch_coeffs[4][2][5] = 1
    patch_coeffs[4][2][6] = 0
    patch_coeffs[4][2][7] = -0.3e1 / 0.2e1
    patch_coeffs[4][2][8] = 0
    patch_coeffs[4][2][9] = 4
    patch_coeffs[4][2][10] = 0
    patch_coeffs[4][2][11] = -2
    patch_coeffs[4][3][0] = -4
    patch_coeffs[4][3][1] = -2
    patch_coeffs[4][3][2] = 6
    patch_coeffs[4][3][3] = -1
    patch_coeffs[4][3][4] = 0
    patch_coeffs[4][3][5] = -1
    patch_coeffs[4][3][6] = 0.1e1 / 0.2e1
    patch_coeffs[4][3][7] = 0
    patch_coeffs[4][3][8] = 0.3e1 / 0.2e1
    patch_coeffs[4][3][9] = -4
    patch_coeffs[4][3][10] = 2
    patch_coeffs[4][3][11] = 0
    patch_coeffs[4][4][0] = -1
    patch_coeffs[4][4][1] = -2
    patch_coeffs[4][4][2] = 3
    patch_coeffs[4][4][3] = -0.1e1 / 0.4e1
    patch_coeffs[4][4][4] = 0
    patch_coeffs[4][4][5] = -0.7e1 / 0.4e1
    patch_coeffs[4][4][6] = 0.5e1 / 0.4e1
    patch_coeffs[4][4][7] = 0
    patch_coeffs[4][4][8] = 0.3e1 / 0.4e1
    patch_coeffs[4][4][9] = -3
    patch_coeffs[4][4][10] = 1
    patch_coeffs[4][4][11] = 0
    patch_coeffs[4][5][0] = -1
    patch_coeffs[4][5][1] = -2
    patch_coeffs[4][5][2] = 3
    patch_coeffs[4][5][3] = -0.7e1 / 0.4e1
    patch_coeffs[4][5][4] = 0.3e1 / 0.2e1
    patch_coeffs[4][5][5] = -0.1e1 / 0.4e1
    patch_coeffs[4][5][6] = -0.1e1 / 0.4e1
    patch_coeffs[4][5][7] = 0.3e1 / 0.2e1
    patch_coeffs[4][5][8] = -0.3e1 / 0.4e1
    patch_coeffs[4][5][9] = -3
    patch_coeffs[4][5][10] = -1
    patch_coeffs[4][5][11] = 2
    patch_coeffs[5][0][0] = 0
    patch_coeffs[5][0][1] = -1
    patch_coeffs[5][0][2] = 2
    patch_coeffs[5][0][3] = -0.3e1 / 0.8e1
    patch_coeffs[5][0][4] = 0.3e1 / 0.8e1
    patch_coeffs[5][0][5] = 0.1e1 / 0.8e1
    patch_coeffs[5][0][6] = -0.3e1 / 0.8e1
    patch_coeffs[5][0][7] = 0.3e1 / 0.8e1
    patch_coeffs[5][0][8] = 0.1e1 / 0.8e1
    patch_coeffs[5][0][9] = -0.1e1 / 0.2e1
    patch_coeffs[5][0][10] = -0.1e1 / 0.2e1
    patch_coeffs[5][0][11] = 0.1e1 / 0.2e1
    patch_coeffs[5][1][0] = 0
    patch_coeffs[5][1][1] = 2
    patch_coeffs[5][1][2] = -2
    patch_coeffs[5][1][3] = 0
    patch_coeffs[5][1][4] = 0
    patch_coeffs[5][1][5] = -1
    patch_coeffs[5][1][6] = 0.3e1 / 0.2e1
    patch_coeffs[5][1][7] = 0
    patch_coeffs[5][1][8] = -0.1e1 / 0.2e1
    patch_coeffs[5][1][9] = 0
    patch_coeffs[5][1][10] = 2
    patch_coeffs[5][1][11] = 0
    patch_coeffs[5][2][0] = 0
    patch_coeffs[5][2][1] = 4
    patch_coeffs[5][2][2] = -4
    patch_coeffs[5][2][3] = 0.3e1 / 0.2e1
    patch_coeffs[5][2][4] = -0.3e1 / 0.2e1
    patch_coeffs[5][2][5] = -0.1e1 / 0.2e1
    patch_coeffs[5][2][6] = 0.3e1 / 0.2e1
    patch_coeffs[5][2][7] = -0.3e1 / 0.2e1
    patch_coeffs[5][2][8] = 0.1e1 / 0.2e1
    patch_coeffs[5][2][9] = 2
    patch_coeffs[5][2][10] = 2
    patch_coeffs[5][2][11] = -2
    patch_coeffs[5][3][0] = 0
    patch_coeffs[5][3][1] = -2
    patch_coeffs[5][3][2] = 2
    patch_coeffs[5][3][3] = 0
    patch_coeffs[5][3][4] = 0
    patch_coeffs[5][3][5] = 2
    patch_coeffs[5][3][6] = -0.5e1 / 0.2e1
    patch_coeffs[5][3][7] = 0
    patch_coeffs[5][3][8] = 0.1e1 / 0.2e1
    patch_coeffs[5][3][9] = 0
    patch_coeffs[5][3][10] = -2
    patch_coeffs[5][3][11] = 0
    patch_coeffs[5][4][0] = 3
    patch_coeffs[5][4][1] = -2
    patch_coeffs[5][4][2] = -1
    patch_coeffs[5][4][3] = 0.3e1 / 0.4e1
    patch_coeffs[5][4][4] = 0
    patch_coeffs[5][4][5] = 0.5e1 / 0.4e1
    patch_coeffs[5][4][6] = -0.7e1 / 0.4e1
    patch_coeffs[5][4][7] = 0
    patch_coeffs[5][4][8] = -0.1e1 / 0.4e1
    patch_coeffs[5][4][9] = 1
    patch_coeffs[5][4][10] = -3
    patch_coeffs[5][4][11] = 0
    patch_coeffs[5][5][0] = 0
    patch_coeffs[5][5][1] = -2
    patch_coeffs[5][5][2] = 2
    patch_coeffs[5][5][3] = -0.3e1 / 0.2e1
    patch_coeffs[5][5][4] = 0.3e1 / 0.2e1
    patch_coeffs[5][5][5] = 0.1e1 / 0.2e1
    patch_coeffs[5][5][6] = -1
    patch_coeffs[5][5][7] = 0.3e1 / 0.2e1
    patch_coeffs[5][5][8] = -1
    patch_coeffs[5][5][9] = -2
    patch_coeffs[5][5][10] = -2
    patch_coeffs[5][5][11] = 2
    patch_coeffs[6][0][0] = -1
    patch_coeffs[6][0][1] = -1
    patch_coeffs[6][0][2] = 3
    patch_coeffs[6][0][3] = -0.3e1 / 0.4e1
    patch_coeffs[6][0][4] = 0.1e1 / 0.4e1
    patch_coeffs[6][0][5] = -0.1e1 / 0.4e1
    patch_coeffs[6][0][6] = 0
    patch_coeffs[6][0][7] = 0.3e1 / 0.4e1
    patch_coeffs[6][0][8] = 0
    patch_coeffs[6][0][9] = -1
    patch_coeffs[6][0][10] = 0
    patch_coeffs[6][0][11] = 1
    patch_coeffs[6][1][0] = 4
    patch_coeffs[6][1][1] = 2
    patch_coeffs[6][1][2] = -6
    patch_coeffs[6][1][3] = 0.3e1 / 0.2e1
    patch_coeffs[6][1][4] = 0.1e1 / 0.2e1
    patch_coeffs[6][1][5] = 0.1e1 / 0.2e1
    patch_coeffs[6][1][6] = 0
    patch_coeffs[6][1][7] = -0.3e1 / 0.2e1
    patch_coeffs[6][1][8] = 0
    patch_coeffs[6][1][9] = 2
    patch_coeffs[6][1][10] = 0
    patch_coeffs[6][1][11] = -2
    patch_coeffs[6][2][0] = 2
    patch_coeffs[6][2][1] = 4
    patch_coeffs[6][2][2] = -6
    patch_coeffs[6][2][3] = 2
    patch_coeffs[6][2][4] = -0.3e1 / 0.2e1
    patch_coeffs[6][2][5] = 1
    patch_coeffs[6][2][6] = 0
    patch_coeffs[6][2][7] = -0.3e1 / 0.2e1
    patch_coeffs[6][2][8] = 0
    patch_coeffs[6][2][9] = 4
    patch_coeffs[6][2][10] = 0
    patch_coeffs[6][2][11] = -2
    patch_coeffs[6][3][0] = -2
    patch_coeffs[6][3][1] = -4
    patch_coeffs[6][3][2] = 6
    patch_coeffs[6][3][3] = -1
    patch_coeffs[6][3][4] = 0.1e1 / 0.2e1
    patch_coeffs[6][3][5] = -1
    patch_coeffs[6][3][6] = 0
    patch_coeffs[6][3][7] = 0.3e1 / 0.2e1
    patch_coeffs[6][3][8] = 0
    patch_coeffs[6][3][9] = -4
    patch_coeffs[6][3][10] = 0
    patch_coeffs[6][3][11] = 2
    patch_coeffs[6][4][0] = -2
    patch_coeffs[6][4][1] = -1
    patch_coeffs[6][4][2] = 3
    patch_coeffs[6][4][3] = -0.3e1 / 0.4e1
    patch_coeffs[6][4][4] = -0.3e1 / 0.4e1
    patch_coeffs[6][4][5] = -0.1e1 / 0.4e1
    patch_coeffs[6][4][6] = 0
    patch_coeffs[6][4][7] = 0.3e1 / 0.4e1
    patch_coeffs[6][4][8] = 0
    patch_coeffs[6][4][9] = -1
    patch_coeffs[6][4][10] = 0
    patch_coeffs[6][4][11] = 1
    patch_coeffs[6][5][0] = -2
    patch_coeffs[6][5][1] = -1
    patch_coeffs[6][5][2] = 3
    patch_coeffs[6][5][3] = -0.7e1 / 0.4e1
    patch_coeffs[6][5][4] = 0.5e1 / 0.4e1
    patch_coeffs[6][5][5] = -0.1e1 / 0.4e1
    patch_coeffs[6][5][6] = 0
    patch_coeffs[6][5][7] = 0.3e1 / 0.4e1
    patch_coeffs[6][5][8] = 0
    patch_coeffs[6][5][9] = -3
    patch_coeffs[6][5][10] = 0
    patch_coeffs[6][5][11] = 1
    patch_coeffs[7][0][0] = -1
    patch_coeffs[7][0][1] = -1
    patch_coeffs[7][0][2] = 3
    patch_coeffs[7][0][3] = -0.1e1 / 0.4e1
    patch_coeffs[7][0][4] = 0
    patch_coeffs[7][0][5] = -0.3e1 / 0.4e1
    patch_coeffs[7][0][6] = 0.1e1 / 0.4e1
    patch_coeffs[7][0][7] = 0
    patch_coeffs[7][0][8] = 0.3e1 / 0.4e1
    patch_coeffs[7][0][9] = -1
    patch_coeffs[7][0][10] = 1
    patch_coeffs[7][0][11] = 0
    patch_coeffs[7][1][0] = 4
    patch_coeffs[7][1][1] = 2
    patch_coeffs[7][1][2] = -6
    patch_coeffs[7][1][3] = 1
    patch_coeffs[7][1][4] = 0
    patch_coeffs[7][1][5] = 2
    patch_coeffs[7][1][6] = -0.3e1 / 0.2e1
    patch_coeffs[7][1][7] = 0
    patch_coeffs[7][1][8] = -0.3e1 / 0.2e1
    patch_coeffs[7][1][9] = 4
    patch_coeffs[7][1][10] = -2
    patch_coeffs[7][1][11] = 0
    patch_coeffs[7][2][0] = 2
    patch_coeffs[7][2][1] = 4
    patch_coeffs[7][2][2] = -6
    patch_coeffs[7][2][3] = 0.1e1 / 0.2e1
    patch_coeffs[7][2][4] = 0
    patch_coeffs[7][2][5] = 0.3e1 / 0.2e1
    patch_coeffs[7][2][6] = 0.1e1 / 0.2e1
    patch_coeffs[7][2][7] = 0
    patch_coeffs[7][2][8] = -0.3e1 / 0.2e1
    patch_coeffs[7][2][9] = 2
    patch_coeffs[7][2][10] = -2
    patch_coeffs[7][2][11] = 0
    patch_coeffs[7][3][0] = -4
    patch_coeffs[7][3][1] = -2
    patch_coeffs[7][3][2] = 6
    patch_coeffs[7][3][3] = -1
    patch_coeffs[7][3][4] = 0
    patch_coeffs[7][3][5] = -1
    patch_coeffs[7][3][6] = 0.1e1 / 0.2e1
    patch_coeffs[7][3][7] = 0
    patch_coeffs[7][3][8] = 0.3e1 / 0.2e1
    patch_coeffs[7][3][9] = -4
    patch_coeffs[7][3][10] = 2
    patch_coeffs[7][3][11] = 0
    patch_coeffs[7][4][0] = -1
    patch_coeffs[7][4][1] = -2
    patch_coeffs[7][4][2] = 3
    patch_coeffs[7][4][3] = -0.1e1 / 0.4e1
    patch_coeffs[7][4][4] = 0
    patch_coeffs[7][4][5] = -0.7e1 / 0.4e1
    patch_coeffs[7][4][6] = 0.5e1 / 0.4e1
    patch_coeffs[7][4][7] = 0
    patch_coeffs[7][4][8] = 0.3e1 / 0.4e1
    patch_coeffs[7][4][9] = -3
    patch_coeffs[7][4][10] = 1
    patch_coeffs[7][4][11] = 0
    patch_coeffs[7][5][0] = -1
    patch_coeffs[7][5][1] = -2
    patch_coeffs[7][5][2] = 3
    patch_coeffs[7][5][3] = -0.1e1 / 0.4e1
    patch_coeffs[7][5][4] = 0
    patch_coeffs[7][5][5] = -0.3e1 / 0.4e1
    patch_coeffs[7][5][6] = -0.3e1 / 0.4e1
    patch_coeffs[7][5][7] = 0
    patch_coeffs[7][5][8] = 0.3e1 / 0.4e1
    patch_coeffs[7][5][9] = -1
    patch_coeffs[7][5][10] = 1
    patch_coeffs[7][5][11] = 0
    patch_coeffs[8][0][0] = 0
    patch_coeffs[8][0][1] = -1
    patch_coeffs[8][0][2] = 2
    patch_coeffs[8][0][3] = 0
    patch_coeffs[8][0][4] = 0
    patch_coeffs[8][0][5] = 0
    patch_coeffs[8][0][6] = -0.1e1 / 0.2e1
    patch_coeffs[8][0][7] = 0
    patch_coeffs[8][0][8] = 0.1e1 / 0.2e1
    patch_coeffs[8][0][9] = 0
    patch_coeffs[8][0][10] = 0
    patch_coeffs[8][0][11] = 0
    patch_coeffs[8][1][0] = 0
    patch_coeffs[8][1][1] = 2
    patch_coeffs[8][1][2] = -2
    patch_coeffs[8][1][3] = 0
    patch_coeffs[8][1][4] = 0
    patch_coeffs[8][1][5] = -1
    patch_coeffs[8][1][6] = 0.3e1 / 0.2e1
    patch_coeffs[8][1][7] = 0
    patch_coeffs[8][1][8] = -0.1e1 / 0.2e1
    patch_coeffs[8][1][9] = 0
    patch_coeffs[8][1][10] = 2
    patch_coeffs[8][1][11] = 0
    patch_coeffs[8][2][0] = 0
    patch_coeffs[8][2][1] = 4
    patch_coeffs[8][2][2] = -4
    patch_coeffs[8][2][3] = 0
    patch_coeffs[8][2][4] = 0
    patch_coeffs[8][2][5] = 0
    patch_coeffs[8][2][6] = 2
    patch_coeffs[8][2][7] = 0
    patch_coeffs[8][2][8] = -1
    patch_coeffs[8][2][9] = 0
    patch_coeffs[8][2][10] = 0
    patch_coeffs[8][2][11] = 0
    patch_coeffs[8][3][0] = 0
    patch_coeffs[8][3][1] = -2
    patch_coeffs[8][3][2] = 2
    patch_coeffs[8][3][3] = 0
    patch_coeffs[8][3][4] = 0
    patch_coeffs[8][3][5] = 2
    patch_coeffs[8][3][6] = -0.5e1 / 0.2e1
    patch_coeffs[8][3][7] = 0
    patch_coeffs[8][3][8] = 0.1e1 / 0.2e1
    patch_coeffs[8][3][9] = 0
    patch_coeffs[8][3][10] = -2
    patch_coeffs[8][3][11] = 0
    patch_coeffs[8][4][0] = 3
    patch_coeffs[8][4][1] = -2
    patch_coeffs[8][4][2] = -1
    patch_coeffs[8][4][3] = 0.3e1 / 0.4e1
    patch_coeffs[8][4][4] = 0
    patch_coeffs[8][4][5] = 0.5e1 / 0.4e1
    patch_coeffs[8][4][6] = -0.7e1 / 0.4e1
    patch_coeffs[8][4][7] = 0
    patch_coeffs[8][4][8] = -0.1e1 / 0.4e1
    patch_coeffs[8][4][9] = 1
    patch_coeffs[8][4][10] = -3
    patch_coeffs[8][4][11] = 0
    patch_coeffs[8][5][0] = 0
    patch_coeffs[8][5][1] = -2
    patch_coeffs[8][5][2] = 2
    patch_coeffs[8][5][3] = 0
    patch_coeffs[8][5][4] = 0
    patch_coeffs[8][5][5] = 0
    patch_coeffs[8][5][6] = -0.3e1 / 0.2e1
    patch_coeffs[8][5][7] = 0
    patch_coeffs[8][5][8] = 0.1e1 / 0.2e1
    patch_coeffs[8][5][9] = 0
    patch_coeffs[8][5][10] = 0
    patch_coeffs[8][5][11] = 0
    patch_coeffs[9][0][0] = 0
    patch_coeffs[9][0][1] = 0
    patch_coeffs[9][0][2] = 1
    patch_coeffs[9][0][3] = 0
    patch_coeffs[9][0][4] = 0
    patch_coeffs[9][0][5] = 0
    patch_coeffs[9][0][6] = 0
    patch_coeffs[9][0][7] = 0
    patch_coeffs[9][0][8] = 0
    patch_coeffs[9][0][9] = 0
    patch_coeffs[9][0][10] = 0
    patch_coeffs[9][0][11] = 0
    patch_coeffs[9][1][0] = 0
    patch_coeffs[9][1][1] = 0
    patch_coeffs[9][1][2] = 0
    patch_coeffs[9][1][3] = 0
    patch_coeffs[9][1][4] = 0
    patch_coeffs[9][1][5] = 0
    patch_coeffs[9][1][6] = 0
    patch_coeffs[9][1][7] = 1
    patch_coeffs[9][1][8] = 0
    patch_coeffs[9][1][9] = 0
    patch_coeffs[9][1][10] = 0
    patch_coeffs[9][1][11] = 0
    patch_coeffs[9][2][0] = 0
    patch_coeffs[9][2][1] = 0
    patch_coeffs[9][2][2] = 0
    patch_coeffs[9][2][3] = 0
    patch_coeffs[9][2][4] = 0
    patch_coeffs[9][2][5] = 0
    patch_coeffs[9][2][6] = 0
    patch_coeffs[9][2][7] = 0
    patch_coeffs[9][2][8] = 1
    patch_coeffs[9][2][9] = 0
    patch_coeffs[9][2][10] = 0
    patch_coeffs[9][2][11] = 0
    patch_coeffs[9][3][0] = 0
    patch_coeffs[9][3][1] = 2
    patch_coeffs[9][3][2] = -2
    patch_coeffs[9][3][3] = 0
    patch_coeffs[9][3][4] = 0
    patch_coeffs[9][3][5] = 0
    patch_coeffs[9][3][6] = 0.1e1 / 0.2e1
    patch_coeffs[9][3][7] = -2
    patch_coeffs[9][3][8] = -0.1e1 / 0.2e1
    patch_coeffs[9][3][9] = 0
    patch_coeffs[9][3][10] = 2
    patch_coeffs[9][3][11] = 0
    patch_coeffs[9][4][0] = 3
    patch_coeffs[9][4][1] = -1
    patch_coeffs[9][4][2] = -2
    patch_coeffs[9][4][3] = 0
    patch_coeffs[9][4][4] = 0.3e1 / 0.4e1
    patch_coeffs[9][4][5] = 0
    patch_coeffs[9][4][6] = -0.1e1 / 0.4e1
    patch_coeffs[9][4][7] = -0.3e1 / 0.4e1
    patch_coeffs[9][4][8] = -0.3e1 / 0.4e1
    patch_coeffs[9][4][9] = 0
    patch_coeffs[9][4][10] = -1
    patch_coeffs[9][4][11] = 1
    patch_coeffs[9][5][0] = 0
    patch_coeffs[9][5][1] = 2
    patch_coeffs[9][5][2] = -2
    patch_coeffs[9][5][3] = 0
    patch_coeffs[9][5][4] = 0
    patch_coeffs[9][5][5] = 0
    patch_coeffs[9][5][6] = 0.1e1 / 0.2e1
    patch_coeffs[9][5][7] = 0
    patch_coeffs[9][5][8] = -0.3e1 / 0.2e1
    patch_coeffs[9][5][9] = 0
    patch_coeffs[9][5][10] = 0
    patch_coeffs[9][5][11] = 0
    patch_coeffs[10][0][0] = 0
    patch_coeffs[10][0][1] = 0
    patch_coeffs[10][0][2] = 1
    patch_coeffs[10][0][3] = 0
    patch_coeffs[10][0][4] = 0
    patch_coeffs[10][0][5] = 0
    patch_coeffs[10][0][6] = 0
    patch_coeffs[10][0][7] = 0
    patch_coeffs[10][0][8] = 0
    patch_coeffs[10][0][9] = 0
    patch_coeffs[10][0][10] = 0
    patch_coeffs[10][0][11] = 0
    patch_coeffs[10][1][0] = 0
    patch_coeffs[10][1][1] = 0
    patch_coeffs[10][1][2] = 0
    patch_coeffs[10][1][3] = 0
    patch_coeffs[10][1][4] = 0
    patch_coeffs[10][1][5] = 0
    patch_coeffs[10][1][6] = 0
    patch_coeffs[10][1][7] = 1
    patch_coeffs[10][1][8] = 0
    patch_coeffs[10][1][9] = 0
    patch_coeffs[10][1][10] = 0
    patch_coeffs[10][1][11] = 0
    patch_coeffs[10][2][0] = 0
    patch_coeffs[10][2][1] = 0
    patch_coeffs[10][2][2] = 0
    patch_coeffs[10][2][3] = 0
    patch_coeffs[10][2][4] = 0
    patch_coeffs[10][2][5] = 0
    patch_coeffs[10][2][6] = 0
    patch_coeffs[10][2][7] = 0
    patch_coeffs[10][2][8] = 1
    patch_coeffs[10][2][9] = 0
    patch_coeffs[10][2][10] = 0
    patch_coeffs[10][2][11] = 0
    patch_coeffs[10][3][0] = 2
    patch_coeffs[10][3][1] = 0
    patch_coeffs[10][3][2] = -2
    patch_coeffs[10][3][3] = 0
    patch_coeffs[10][3][4] = 0.1e1 / 0.2e1
    patch_coeffs[10][3][5] = 0
    patch_coeffs[10][3][6] = 0
    patch_coeffs[10][3][7] = -0.1e1 / 0.2e1
    patch_coeffs[10][3][8] = -2
    patch_coeffs[10][3][9] = 0
    patch_coeffs[10][3][10] = 0
    patch_coeffs[10][3][11] = 2
    patch_coeffs[10][4][0] = 2
    patch_coeffs[10][4][1] = 0
    patch_coeffs[10][4][2] = -2
    patch_coeffs[10][4][3] = 0
    patch_coeffs[10][4][4] = 0.1e1 / 0.2e1
    patch_coeffs[10][4][5] = 0
    patch_coeffs[10][4][6] = 0
    patch_coeffs[10][4][7] = -0.3e1 / 0.2e1
    patch_coeffs[10][4][8] = 0
    patch_coeffs[10][4][9] = 0
    patch_coeffs[10][4][10] = 0
    patch_coeffs[10][4][11] = 0
    patch_coeffs[10][5][0] = -1
    patch_coeffs[10][5][1] = 3
    patch_coeffs[10][5][2] = -2
    patch_coeffs[10][5][3] = 0
    patch_coeffs[10][5][4] = -0.1e1 / 0.4e1
    patch_coeffs[10][5][5] = 0
    patch_coeffs[10][5][6] = 0.3e1 / 0.4e1
    patch_coeffs[10][5][7] = -0.3e1 / 0.4e1
    patch_coeffs[10][5][8] = -0.3e1 / 0.4e1
    patch_coeffs[10][5][9] = 0
    patch_coeffs[10][5][10] = 1
    patch_coeffs[10][5][11] = -1
    patch_coeffs[11][0][0] = -1
    patch_coeffs[11][0][1] = 0
    patch_coeffs[11][0][2] = 2
    patch_coeffs[11][0][3] = 0
    patch_coeffs[11][0][4] = -0.1e1 / 0.2e1
    patch_coeffs[11][0][5] = 0
    patch_coeffs[11][0][6] = 0
    patch_coeffs[11][0][7] = 0.1e1 / 0.2e1
    patch_coeffs[11][0][8] = 0
    patch_coeffs[11][0][9] = 0
    patch_coeffs[11][0][10] = 0
    patch_coeffs[11][0][11] = 0
    patch_coeffs[11][1][0] = 4
    patch_coeffs[11][1][1] = 0
    patch_coeffs[11][1][2] = -4
    patch_coeffs[11][1][3] = 0
    patch_coeffs[11][1][4] = 2
    patch_coeffs[11][1][5] = 0
    patch_coeffs[11][1][6] = 0
    patch_coeffs[11][1][7] = -1
    patch_coeffs[11][1][8] = 0
    patch_coeffs[11][1][9] = 0
    patch_coeffs[11][1][10] = 0
    patch_coeffs[11][1][11] = 0
    patch_coeffs[11][2][0] = 2
    patch_coeffs[11][2][1] = 0
    patch_coeffs[11][2][2] = -2
    patch_coeffs[11][2][3] = -1
    patch_coeffs[11][2][4] = 0.3e1 / 0.2e1
    patch_coeffs[11][2][5] = 0
    patch_coeffs[11][2][6] = 0
    patch_coeffs[11][2][7] = -0.1e1 / 0.2e1
    patch_coeffs[11][2][8] = 0
    patch_coeffs[11][2][9] = 0
    patch_coeffs[11][2][10] = 0
    patch_coeffs[11][2][11] = 2
    patch_coeffs[11][3][0] = -2
    patch_coeffs[11][3][1] = 0
    patch_coeffs[11][3][2] = 2
    patch_coeffs[11][3][3] = 2
    patch_coeffs[11][3][4] = -0.5e1 / 0.2e1
    patch_coeffs[11][3][5] = 0
    patch_coeffs[11][3][6] = 0
    patch_coeffs[11][3][7] = 0.1e1 / 0.2e1
    patch_coeffs[11][3][8] = 0
    patch_coeffs[11][3][9] = 0
    patch_coeffs[11][3][10] = 0
    patch_coeffs[11][3][11] = -2
    patch_coeffs[11][4][0] = -2
    patch_coeffs[11][4][1] = 0
    patch_coeffs[11][4][2] = 2
    patch_coeffs[11][4][3] = 0
    patch_coeffs[11][4][4] = -0.3e1 / 0.2e1
    patch_coeffs[11][4][5] = 0
    patch_coeffs[11][4][6] = 0
    patch_coeffs[11][4][7] = 0.1e1 / 0.2e1
    patch_coeffs[11][4][8] = 0
    patch_coeffs[11][4][9] = 0
    patch_coeffs[11][4][10] = 0
    patch_coeffs[11][4][11] = 0
    patch_coeffs[11][5][0] = -2
    patch_coeffs[11][5][1] = 3
    patch_coeffs[11][5][2] = -1
    patch_coeffs[11][5][3] = 0.5e1 / 0.4e1
    patch_coeffs[11][5][4] = -0.7e1 / 0.4e1
    patch_coeffs[11][5][5] = 0.3e1 / 0.4e1
    patch_coeffs[11][5][6] = 0
    patch_coeffs[11][5][7] = -0.1e1 / 0.4e1
    patch_coeffs[11][5][8] = 0
    patch_coeffs[11][5][9] = 1
    patch_coeffs[11][5][10] = 0
    patch_coeffs[11][5][11] = -3

    # Redundant check
    assert patch_coeffs.shape == (12, 6, 12)
    return patch_coeffs
