from collections.abc import Iterator
from dataclasses import dataclass
import sympy as sp
from sympy.physics.paulialgebra import Pauli

# basic Pauli canonicalizer for converting
# sympy expression to {product key: coeff}

OPS = "XYZ"

type Product = dict[str, str]            # site label -> "X" | "Y" | "Z"
type Key = tuple[tuple[str, str], ...]    # hashable, sorted Product
type Terms = dict[Key, sp.Expr]

@dataclass
class QubitRegister:
    # Represents N-qubit site operator used to build Hamiltonians
    N : int = 0

    def X(self, i: int): return Pauli(1, label=f"q{i}")
    def Y(self, i: int): return Pauli(2, label=f"q{i}")
    def Z(self, i: int): return Pauli(3, label=f"q{i}")

def _mul(a: str, b: str) -> tuple[sp.Expr | int, str]:
    # single-site product a*b -> (phase, op)
    if not a: return 1, b
    if not b: return 1, a
    if a == b: return 1, "" # identity
    c = OPS[3 - OPS.index(a) - OPS.index(b)]
    # reduce XY=iZ, YZ=iX, ZX=iY; anticyclic gets -i
    return (sp.I if (OPS.index(b) - OPS.index(a)) % 3 == 1 else -sp.I), c

def _factors(f: sp.Expr) -> Iterator[Pauli]:
    # yield Pauli factors of a non-commutative factor, unrolling integer powers
    if f.is_Pow and isinstance(f.base, Pauli) and f.exp.is_Integer and f.exp > 0:
        yield from [f.base] * int(f.exp)
    elif isinstance(f, Pauli):
        yield f
    else:
        raise TypeError(f"non-Pauli operator in Hamiltonian: {f}")

def key(prod: Product) -> Key:
    # hashable canonical form of a product dict
    return tuple(sorted(prod.items()))

def canonicalize(H: sp.Expr) -> Terms:
    # Expand H, reduce same-site products, merge like terms, drop zero terms and identity
    merged: dict[Key, sp.Expr] = {}
    for term in sp.Add.make_args(sp.expand(H)):
        c, nc = term.args_cnc()
        coeff, prod = sp.Mul(*c), {}
        for f in nc:
            for p in _factors(f):
                phase, prod[p.label] = _mul(prod.get(p.label, ""), OPS[p.i - 1])
                coeff *= phase
        prod = {s: op for s, op in prod.items() if op}
        k = key(prod)
        merged[k] = merged.get(k, 0) + coeff
    return {k: c for k, c in ((k, sp.expand(c)) for k, c in merged.items()) if k and c != 0}

def coords(H: Terms, basis: list[Key]) -> list[sp.Expr]:
    # coefficients of H in the Pauli basis; absent products are 0
    return [H.get(m, 0) for m in basis]
