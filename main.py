from target import EvolvePiecewise
from system import System

import sympy as sp
from scipy.optimize import least_squares
import numpy as np

N = 3

# Define target Hamiltonian: simple transverse-field Ising model for one unit-time
H_trgt = EvolvePiecewise(N)
X = sp.symbols(f"X0:{N}")
Z = sp.symbols(f"Z0:{N}")
H_tf = sum(Z[i]*Z[i+1] for i in range(N-1)) + sum(X)
H_trgt.add(H_tf, duration=1.0)

# Define device Hamiltonian: Rydberg atom array
H_sys = System(N)

# Van der Waals pairwise forces (intrinsic dynamics)
x = [H_sys.add_global_var(f"x{i}") for i in range(N)] # global positions
n = [(1 - Z[i]) / 2 for i in range(N)] # operator number
C = 5.42e6 # MHz/um^6
for i in range(N):
    for j in range(i):
        H_sys.H += C / (x[i] - x[j])**6 * n[i] * n[j]

# Laser (control dynamics)
Y = sp.symbols(f"Y0:{N}")
for i in range(N):
    delta, omega, phi = sp.symbols(f"delta{i} omega{i} phi{i}", real=True)
    h = -delta*n[i] + omega/2 * (sp.cos(phi)*X[i] - sp.sin(phi)*Y[i])
    H_sys.add_instruction(h, delta, omega, phi)

ops = [*X, *Y, *Z]
def paulis(H):
    # keys are binary indicators for Pauli ops
    terms = sp.Poly(H, *ops).as_dict()
    terms.pop((0,)*len(ops), None)
    return terms

# match t_k*H_sys = tau_k*H_trgt,k for each segment k
c_sys = paulis(H_sys.system())
c_trgt = [paulis(h) for h in H_trgt.system()]
basis = sorted(c_sys.keys() | set().union(*c_trgt))

time = sp.Symbol("t", positive=True)
glob, loc = H_sys.globals, list(H_sys.locals())
f = sp.lambdify([*glob, time, *loc], [time*c_sys.get(m,0) for m in basis], "numpy")
Y_trgt = np.array([[step.duration*float(c.get(m,0)) for m in basis]
                   for step, c in zip(H_trgt.terms, c_trgt)])

# p = [globals (shared), (t, locals) per segment]
G, L = len(glob), 1 + len(loc)
def residual(p):
    g, segs = p[:G], p[G:].reshape(-1, L)
    res = [np.asarray(f(*g, s[0],*s[1:]), dtype=float) - y for s, y in zip(segs, Y_trgt)]
    return np.concatenate(res)

p0 = np.r_[8*np.arange(N), np.tile(np.r_[0.1, np.tile([0, 20, 0], N)], len(H_trgt.terms))]
matching = least_squares(residual, p0)
print(matching)
