from hml import EvolvePiecewise
from aais import Device
from solver import solve

import sympy as sp

N = 3

# Define target Hamiltonian: simple transverse-field Ising model for one unit-time
ising = EvolvePiecewise(N)
h = sum(ising.Z(i)*ising.Z(i+1) for i in range(N-1)) + sum(ising.X(i) for i in range(N))

ising.add(h, duration=1.0)

# Define device Hamiltonian: Rydberg atom array
rydberg = Device(N)

# Van der Waals pairwise forces (intrinsic dynamics)
x = [rydberg.add_global_var(f"x{i}", init=1.0*i) for i in range(N)] # global atom positions
n = [(1 - rydberg.Z(i)) / 2 for i in range(N)] # operator number
C = 5.42e6 # MHz/um^6
h = 0
for i in range(N):
    for j in range(i):
        h += C / (x[i] - x[j])**6 * n[i] * n[j]
rydberg.set_sys_ham(h)

# Laser (control dynamics)
h = 0
for i in range(N):
    delta, omega, phi = sp.symbols(f"delta{i} omega{i} phi{i}", real=True)
    h = -delta*n[i] + omega/2 * (sp.cos(phi)*rydberg.X(i) - sp.sin(phi)*rydberg.Y(i))
    rydberg.add_instruction(h, delta, omega, phi)



matching = solve(ising, rydberg)
print(matching)
