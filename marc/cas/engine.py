import sympy as sp
from .residuals import get_residuals
from .energy import get_energy

class CASEngine:
    def __init__(self, json_path, symbol_names):
        self.x = sp.symbols(symbol_names)
        self._residual_exprs = get_residuals(json_path) # FIX: Renamed to avoid collision
        self._energy_expr = get_energy(self._residual_exprs)
        self._grad_expr = [sp.diff(self._energy_expr, xi) for xi in self.x]
        
        # Pre-compile for fast execution
        self._res_funcs = [sp.lambdify(self.x, r, 'numpy') for r in self._residual_exprs] # FIX: Lambdified
        self._energy_func = sp.lambdify(self.x, self._energy_expr, 'numpy')
        self._grad_func = sp.lambdify(self.x, self._grad_expr, 'numpy')

    def residuals(self, x_vals):
        # Now fast and doesn't collide with the instance variable
        return [float(f(*x_vals)) for f in self._res_funcs]

    def energy(self, x_vals):
        return float(self._energy_func(*x_vals))

    def energy_grad(self, x_vals):
        return list(self._grad_func(*x_vals))

    def accepts(self, x_vals, tol=1e-6):
        # Numeric-only gate
        return self.energy(x_vals) < tol