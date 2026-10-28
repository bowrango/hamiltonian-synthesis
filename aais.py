from dataclasses import dataclass, field
import sympy as sp
from pauli import QubitRegister, Terms, canonicalize

# barebones analog abstract instruction set (AAIS)

@dataclass
class Instruction:
    h: sp.Expr
    locals: tuple = ()

@dataclass
class Device(QubitRegister):
    # Represents an N-qubit Hamiltonian defined from intrinsic system dynamics (H, globals)
    # and available control dynamics (instructions)
    _H: sp.Expr = 0
    globals : list = field(default_factory=list)
    instructions : list = field(default_factory=list)
    init : dict = field(default_factory=dict) # initial values for variables

    def locals(self):
        return [p for inst in self.instructions for p in inst.locals]

    # Define system dynamics
    def add_global_var(self, name, init=0.0):
        x = sp.Symbol(name, real=True)
        self.globals.append(x)
        self.init[x] = init
        return x

    def set_sys_ham(self, H):
        self._H= H

    # Define control dynamics
    def add_instruction(self, h, *params, init=None):
        eta = Instruction(h, params)
        self.instructions.append(eta)
        self.init.update(zip(params, init or [0.0]*len(params)))
        return eta

    def expand(self) -> sp.Expr:
        return sp.expand(self._H+ sum(inst.h for inst in self.instructions))

    def paulis(self) -> Terms:
        return canonicalize(self.expand())

    # def hamiltonian(self, indicators):
    #     # indicate active instructions
    #     assert(len(indicators)==len(self.instructions))
    #     H_local = sum(s*inst.h for s,inst in zip(indicators,self.instructions))
    #     return sp.expand(self._H+ H_local)