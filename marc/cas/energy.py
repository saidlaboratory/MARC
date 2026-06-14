import sympy as sp

def get_energy(residuals):
    # Returns the symbolic energy expression
    return sp.Rational(1, 2) * sum([g**2 for g in residuals])