"""ballrain.geometry: the compiled vertical-ray kernel gives exactly the numpy path's result (heights and triangles,
ties included)."""
import numpy as np

import ballrain.geometry as G


def test_compiled_rays_match_numpy_exactly():
    rng = np.random.default_rng(1)
    n = 3000
    A = rng.uniform(-40, 40, (n, 3))
    B = A + rng.normal(0, 3, (n, 3))
    C = A + rng.normal(0, 3, (n, 3))
    A[::5, 1] = B[::5, 1] = C[::5, 1] = 2.0  # flat triangles at one height: ties
    T = G.TriangleSet(A, B, C)
    P = rng.uniform(-45, 45, (8000, 3))
    P[:, 1] = rng.uniform(-10, 40, 8000)
    fast = T.ray_down(P)
    keep = G.HAVE_NUMBA
    try:
        G.HAVE_NUMBA = False
        slow = T.ray_down(P)
    finally:
        G.HAVE_NUMBA = keep
    assert np.array_equal(fast[0], slow[0], equal_nan=True)
    assert np.array_equal(fast[1], slow[1])
    assert (~np.isnan(slow[0])).sum() > 1000
