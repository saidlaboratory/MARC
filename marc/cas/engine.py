import sympy as sp
from .residuals import get_residuals
from .energy import get_energy

class CASEngine:
    def __init__(self, json_path, symbol_names):
        self.x = sp.symbols(symbol_names)
        self.residuals = get_residuals(json_path)
        self.energy_expr = get_energy(self.residuals)
        self.grad_expr = [sp.diff(self.energy_expr, xi) for xi in self.x]
        
        # Pre-compile for fast execution
        self._energy_func = sp.lambdify(self.x, self.energy_expr, 'numpy')
        self._grad_func = sp.lambdify(self.x, self.grad_expr, 'numpy')

    def residuals(self, x_vals):
        return [float(r.subs(dict(zip(self.x, x_vals)))) for r in self.residuals]

    def energy(self, x_vals):
        return float(self._energy_func(*x_vals))

    def energy_grad(self, x_vals):
        return list(self._grad_func(*x_vals))

    def accepts(self, x_vals, tol=1e-6):
        # Numeric-only gate
        return self.energy(x_vals) < tol