import numpy as np

V_coeffs = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
assert V_coeffs.shape == (3, 3)
what = V_coeffs[[2], :]

print(what.shape
      )

print(what)
