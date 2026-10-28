import sympy as sp
import numpy as np
from scipy.optimize import least_squares
from pauli import coords

def solve(target, device, t0=0.1):
    # Match t_k*H_dev = tau_k*H_trgt,k for each time-segment k
    H_dev = device.paulis()                               # {Key: Expr}
    segments = target.paulis()                            # [({Key: Expr}, duration)]
    basis = sorted(H_dev.keys() | {k for h, _ in segments for k in h})

    # globals variables (shared)
    glob = device.globals
    # local variables
    t = sp.Symbol("t", positive=True)
    loc = [t, *device.locals()]
    G, L, K = len(glob), len(loc), len(segments)
    init = {t: t0, **device.init}

    f = sp.lambdify([glob, loc], [t*c for c in coords(H_dev, basis)], "numpy")

    def lhs(p):
        g = p[:G]
        l = p[G:].reshape(K, L)
        # broadcast scalar constant entries (K,)
        return np.stack(np.broadcast_arrays(*f(g, l.T)), axis=-1)   # (K, B)

    rhs = np.array([[dur*c for c in coords(h, basis)]
                              for h, dur in segments], dtype=float)  # (K, B)

    def residual(p):                                      # (K*B,)
        return (lhs(p) - rhs).ravel()

    p0 = np.r_[[init[v] for v in glob], np.tile([init[v] for v in loc], K)]
    return least_squares(residual, p0)
