from dataclasses import dataclass, field
import sympy as sp

# barebones analog abstract instruction set (AAIS)

@dataclass
class Term:
    h: sp.Expr
    duration: float

@dataclass
class EvolvePiecewise:
    # Represents an N-qubit Hamiltonian defined with piecewise-constant terms
    # h1, 0 <= t <= dur1
    # h2, dur1 <= t <= dur1+dur2
    N : int = 0
    terms : list = field(default_factory=list)

    def add(self, h : sp.Expr, duration : float):
        self.terms.append(Term(h, duration))

    def system(self):
        # one expanded Hamiltonian per segment
        return [sp.expand(step.h) for step in self.terms]