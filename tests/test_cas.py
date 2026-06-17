from marc.cas.engine import CASEngine
import pytest

def test_cas_engine_solution():
    # Initialize
    engine = CASEngine('marc/data/examples/two_equations.json', 'x y')
    
    # Test solution (x=2, y=1)
    test_solution = [2.0, 1.0]
    
    # Assertions
    assert engine.energy(test_solution) == pytest.approx(0.0, abs=1e-6)
    assert engine.energy_grad(test_solution) == pytest.approx([0.0, 0.0], abs=1e-6)