from marc.cas.checker import Checker
from marc.eval.structure_eval import (
    RunRecord,
    TOYS,
    auxiliary_usage_rate,
    run_h2_suite,
    solve_fixed,
    solve_rate,
    solve_structure,
    toy_bilinear_product,
    toy_quadratic_link,
    toy_sum_product,
)


def test_toy_fixed_graphs_accept_their_own_solution():
    """Every toy's stated (x*, y*, ...) solution must satisfy the *fixed* graph — the
    invariant the P0 generator enforces for the other templates (marc/data/generator.py)."""
    checker = Checker()
    for maker in (toy_sum_product, toy_bilinear_product, toy_quadratic_link):
        for seed in range(10):
            problem = maker(seed)
            x_star = [problem.solution[v.id] for v in problem.fixed_graph.variables]
            assert checker.accepts(problem.fixed_graph, x_star), (problem.description, x_star)


def test_aux_graph_is_a_superset_of_the_fixed_graph():
    for maker in TOYS.values():
        problem = maker(0)
        fixed_ids = {v.id for v in problem.fixed_graph.variables}
        aux_ids = {v.id for v in problem.aux_graph.variables}
        assert fixed_ids < aux_ids
        assert problem.aux_var in aux_ids - fixed_ids

        fixed_factor_ids = {f.id for f in problem.fixed_graph.factors}
        aux_factor_ids = {f.id for f in problem.aux_graph.factors}
        assert fixed_factor_ids <= aux_factor_ids  # aux only adds factors, never removes


def test_aux_graph_solution_still_satisfies_original_factors():
    """Solving the augmented graph must never change the original solution set: the
    base variables of any assignment accepted on aux_graph also satisfy fixed_graph."""
    checker = Checker()
    problem = toy_sum_product(seed=2)
    n_base = len(problem.fixed_graph.variables)
    record = solve_structure(problem, k=10, steps=300, seed=42)
    if record.accepted:
        assert checker.accepts(problem.fixed_graph, record.x_base[:n_base])


def test_solve_fixed_and_solve_structure_return_run_records():
    problem = toy_sum_product(seed=1)
    fixed = solve_fixed(problem, k=4, steps=200, seed=0)
    structure = solve_structure(problem, k=4, steps=200, seed=1)
    assert isinstance(fixed, RunRecord)
    assert isinstance(structure, RunRecord)
    assert fixed.used_aux is False  # the fixed model never has access to the aux slot


def test_auxiliary_usage_rate_only_counts_solved_runs():
    solved_with_aux = RunRecord("t", 0, True, True, [0.0], 0.0)
    solved_without_aux = RunRecord("t", 0, True, False, [0.0], 0.0)
    unsolved_with_aux = RunRecord("t", 0, False, True, [0.0], 1.0)

    assert auxiliary_usage_rate([solved_with_aux, solved_without_aux]) == 0.5
    assert auxiliary_usage_rate([unsolved_with_aux]) == 0.0  # nothing solved
    assert auxiliary_usage_rate([]) == 0.0


def test_solve_rate_requires_nonempty():
    import pytest

    with pytest.raises(ValueError):
        solve_rate([])


def test_run_h2_suite_summary_shape():
    summary, fixed_records, structure_records = run_h2_suite(
        toy_names=["sum_product"], n_instances=3, k=4, steps=200, base_seed=0
    )
    assert set(summary["toys"].keys()) == {"sum_product"}
    toy_summary = summary["toys"]["sum_product"]
    assert toy_summary["n_instances"] == 3
    assert 0.0 <= toy_summary["fixed_solve_rate"] <= 1.0
    assert 0.0 <= toy_summary["structure_solve_rate"] <= 1.0
    assert 0.0 <= toy_summary["auxiliary_usage_rate"] <= 1.0
    assert len(fixed_records) == 3
    assert len(structure_records) == 3


def test_run_record_to_dict_roundtrip_shape():
    record = RunRecord("sum_product", 0, True, True, [1.0, 2.0], 0.0, "desc")
    d = record.to_dict(model="structure")
    assert d["toy"] == "sum_product"
    assert d["model"] == "structure"
    assert d["used_aux"] is True
    assert d["x_base"] == [1.0, 2.0]
