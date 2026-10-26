from target import EvolvePiecewise
from system import System

import sympy as sp

N = 3

# Define target Hamiltonian: simple transverse-field Ising model for one unit-time
H_trgt = EvolvePiecewise(N)
X = sp.symbols(f"X0:{N}", communative=False)
Z = sp.symbols(f"Z0:{N}", communative=False)
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
Y = sp.symbols(f"Y0:{N}", communative=False)
for i in range(N):
    delta, omega, phi = sp.symbols(f"delta{i} omega{i} phi{i}", real=True)
    h = (-delta*n[i] + omega/2 * (sp.cos(phi)*X[i] - sp.sin(phi)*Y[i]))
    H_sys.add_instruction(h, delta, omega, phi)
print(H_sys.hamiltonian([1,1,1]))