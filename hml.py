from dataclasses import dataclass, field
import sympy as sp
from pauli import QubitRegister, Terms, canonicalize

# barebones Hamiltonian Modeling Language (HML)

@dataclass
class Term:
    h: sp.Expr
    duration: float

@dataclass
class EvolvePiecewise(QubitRegister):
    # Represents an N-qubit Hamiltonian defined with piecewise-constant terms
    # h1, 0 <= t <= dur1
    # h2, dur1 <= t <= dur1+dur2
    terms : list = field(default_factory=list)

    def add(self, h : sp.Expr, duration : float) -> None:
        self.terms.append(Term(h, duration))

    def expand(self) -> list[sp.Expr]:
        # one expanded Hamiltonian per segment
        return [sp.expand(step.h) for step in self.terms]

    def paulis(self) -> list[tuple[Terms, float]]:
        # one ({product key: coeff}, duration) per segment
        return [(canonicalize(step.h), step.duration) for step in self.terms]
