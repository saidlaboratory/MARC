# tests/test_corruption.py
import pytest
from marc.cas.engine import CASEngine

def test_corruption_energy_increase():
    # 1. Initialize
    engine = CASEngine('marc/data/examples/two_equations.json', 'x y')
    
    # 2. Known solution (E=0)
    x_star = [2.0, 1.0]
    assert engine.energy(x_star) == pytest.approx(0.0, abs=1e-6)
    
    # 3. Apply corruption (Add noise to the solution)
    corruption = [0.1, -0.1]
    x_t = [x_star[0] + corruption[0], x_star[1] + corruption[1]]
    
    # 4. Assert that corruption increases energy
    e_t = engine.energy(x_t)
    assert e_t > 0.0, f"Corruption failed: Energy should be > 0, but got {e_t}"