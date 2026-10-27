from dataclasses import dataclass, field
import sympy as sp

# barebones Hamiltonian Modeling Language (HML)

@dataclass
class Instruction:
    h: sp.Expr
    locals: tuple = ()

# @dataclass
# class Layout:
#     mapping : 
#     basis :

@dataclass
class System:
    # Represents an N-qubit Hamiltonian defined from intrinsic system dynamics (H, globals)
    # and available control dynamics (instructions)
    N : int = 0
    H : sp.Expr = 0
    globals : list = field(default_factory=list)
    instructions : list = field(default_factory=list)

    def locals(self):
        return (p for inst in self.instructions for p in inst.locals)

    # Define system dynamics
    def add_global_var(self, name):
        x = sp.Symbol(name, real=True)
        self.globals.append(x)
        return x

    # Define control dynamics
    def add_instruction(self, h, *params):
        eta = Instruction(h, params)
        self.instructions.append(eta)
        return eta

    def system(self):
        return sp.expand(self.H + sum(inst.h for inst in self.instructions))
    
    # def hamiltonian(self, indicators):
    #     # indicate active instructions
    #     assert(len(indicators)==len(self.instructions))
    #     H_local = sum(s*inst.h for s,inst in zip(indicators,self.instructions))
    #     return sp.expand(self.H + H_local)