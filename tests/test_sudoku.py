"""Tests for the 4x4 Sudoku Grover implementation."""

import pytest

from grover_miniproject import sudoku as sd
from grover_miniproject.theory import grover_probability

SOLVED = [[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]]


def blank(cells):
    g = [row[:] for row in SOLVED]
    for r, c in cells:
        g[r][c] = 0
    return g


ONE = blank([(0, 0), (0, 1), (1, 0)])  # unique completion
TWO = blank([(0, 0), (0, 1), (2, 0), (2, 1)])  # two completions
NONE = [[1, 2, 3, 4], [3, 4, 1, 0], [2, 1, 0, 3], [0, 3, 4, 1]]  # none
PUZZLES = [(ONE, 1), (TWO, 2), (NONE, 0)]


def test_checker_on_known_grids():
    assert sd.is_valid_grid(SOLVED)
    bad = [row[:] for row in SOLVED]
    bad[0][0], bad[0][1] = bad[0][1], bad[0][0]  # swap breaks columns/blocks
    assert not sd.is_valid_grid(bad)
    assert not sd.is_valid_grid([[1] * 4 for _ in range(4)])


def test_solution_counts():
    for g, m in PUZZLES:
        assert len(sd.brute_force_solutions(g)) == m


def test_layout_counts():
    s, f, a = sd.layout(ONE)
    assert len(s) == 6 and len(f) == 3 + 3 and len(a) == len(f) - 1


def test_oracle_matches_independent_checker_and_cleans_work_qubits():
    for g, _ in PUZZLES:
        sd.check_oracle(g)  # asserts sign == checker and work qubits restored


def test_oracle_statevector_agrees_on_small_puzzle():
    # Cross-check the classical bit simulation against Qiskit's Statevector.
    sd.check_oracle_statevector(ONE)


def test_inconsistent_clues_rejected():
    g = [row[:] for row in SOLVED]
    g[0][0] = 0
    g[1][1] = 1  # duplicates the 1 in the same block/column neighbours
    with pytest.raises(ValueError):
        sd.sudoku_oracle(g)
    assert sd.sample_solution(g) is None


def test_no_empty_cells_rejected():
    with pytest.raises(ValueError):
        sd.sudoku_oracle(SOLVED)


def test_success_curve_matches_closed_form():
    for g, m in PUZZLES:
        n_total = 4 ** len(sd.empties(g))
        for k, p in enumerate(sd.success_curve(g, 6)):
            assert abs(p - grover_probability(n_total, m, k)) < 1e-12


def test_sampler_returns_valid_grid_or_none():
    for g, m in PUZZLES:
        out = sd.sample_solution(g, max_trials=40, seed=1)
        if m == 0:
            assert out is None
        else:
            assert out is not None and sd.is_valid_grid(out)
            for r in range(4):  # clues preserved
                for c in range(4):
                    assert g[r][c] in (0, out[r][c])


def test_resources_report():
    r = sd.resources(ONE)
    assert r["search_qubits"] == 6 and r["work_qubits"] == 11
    assert r["oracle_depth"] > 0


def solved_grid(block):
    side = block * block
    return [
        [(r * block + r // block + c) % side + 1 for c in range(side)]
        for r in range(side)
    ]


@pytest.mark.parametrize("block", [1, 2, 3, 4, 5])
def test_general_sizes_domain_phases_and_cleanup(block):
    g = solved_grid(block)
    assert sd.is_valid_grid(g)
    g[0][0] = 0
    sd.check_oracle(g)  # all binary codes, including unused digits, and scratch
    sd.check_oracle_statevector(g)
    assert sd.brute_force_solutions(g) == [(1,)]
    n_total = sd.search_size(g)
    for j, p in enumerate(sd.success_curve(g, 4)):
        assert p == pytest.approx(grover_probability(n_total, 1, j))
    report = sd.sample_solution_report(g, max_trials=60, seed=9)
    assert report.solution == solved_grid(block)
    assert report.oracle_calls <= report.oracle_call_budget


@pytest.mark.parametrize("block", [2, 3, 4, 5])
def test_general_pair_checks_and_unsatisfiable_puzzles(block):
    g = solved_grid(block)
    g[0][0] = g[0][1] = 0  # shared unit; exercise wider pair comparisons
    sd.check_oracle(g)
    assert len(sd.brute_force_solutions(g)) == 1
    g = solved_grid(block)
    g[0][0] = 0
    g[1][0] = 1
    g[1][block * block - block] = 0
    assert sd.clues_consistent(g)
    assert not sd.brute_force_solutions(g)
    sd.check_oracle(g)
    report = sd.sample_solution_report(g, seed=8)
    assert report.solution is None
    assert report.trials == sd.trial_budget()
    assert report.oracle_calls <= report.oracle_call_budget


@pytest.mark.parametrize("block", [2, 3, 5])
def test_actual_complex_superposition_phases_and_clean_work(block):
    import numpy as np
    from qiskit.quantum_info import Statevector

    g = solved_grid(block)
    g[0][0] = 0
    circuit = sd.sudoku_oracle(g)
    size = sd.search_size(g)
    rng = np.random.default_rng(4)
    search = rng.normal(size=size) + 1j * rng.normal(size=size)
    search /= np.linalg.norm(search)
    initial = np.zeros(1 << circuit.num_qubits, dtype=complex)
    initial[:size] = search
    expected = initial.copy()
    # Independent clue-preserving solution is digit 1 (binary code 0).
    expected[0] *= -1
    actual = Statevector(initial).evolve(circuit).data
    np.testing.assert_allclose(actual, expected, atol=1e-12)


BAD_GRIDS = [[], [[0, 0], [0, 0]], [[0] * 4] * 3, [[0] * 4, [0] * 3, [0] * 4, [0] * 4]]
for invalid in (5, -1, 1.5, True, "1", None):
    bad = blank([(0, 0)])
    bad[1][1] = invalid
    BAD_GRIDS.append(bad)


@pytest.mark.parametrize("grid", BAD_GRIDS)
def test_malformed_inputs_rejected_before_any_circuit(grid, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("circuit construction happened before input validation")

    monkeypatch.setattr(sd, "QuantumCircuit", forbidden)
    for builder in (
        sd.sudoku_oracle,
        sd.compute_checks_circuit,
        sd.phase_circuit,
        sd.uncompute_checks_circuit,
        sd.grover_block_circuit,
        sd.diffuser,
        sd.resources,
    ):
        with pytest.raises(ValueError):
            builder(grid)
    with pytest.raises(ValueError):
        sd.sample_solution(grid)
    assert not sd.is_valid_grid(grid)


def test_probability_budget_for_every_possible_marked_count():
    import math

    trials = sd.trial_budget()
    assert trials == 9
    assert (7 / 8) ** trials <= 1 / 3
    for size in (2, 4, 16, 32, 64, 256):
        cap = math.isqrt(4 * size - 1) + 1
        for marked in range(1, size + 1):
            p = (
                0.5 * marked / size
                + 0.5
                * sum(grover_probability(size, marked, j) for j in range(cap))
                / cap
            )
            assert p >= 1 / 8 - 1e-12
            assert 1 - (1 - p) ** trials >= 2 / 3


def test_report_tracks_actual_queries_and_budget(monkeypatch):
    class FixedRandom:
        def __init__(self, seed):
            pass

        def randrange(self, stop):
            return stop - 1  # amplified branch, maximum iterations

        def choices(self, population, weights):
            return [0]  # valid unique filling for the one-blank example

    monkeypatch.setattr(sd.random, "Random", FixedRandom)
    g = blank([(0, 0)])
    result = sd.sample_solution_report(g)
    assert result.trials == 1
    assert result.oracle_calls == 3
    assert result.oracle_call_budget == 27
    assert result.failure_bound_if_solvable == (7 / 8) ** 9
    assert sd.resources(g)["oracle_call_budget"] == 27
    assert sd.sample_solution_report(g, max_trials=0).failure_bound_if_solvable == 1
    assert sd.sample_solution(SOLVED) == SOLVED


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_bad_iteration_and_trial_budgets(value):
    with pytest.raises(ValueError):
        sd.grover_circuit(ONE, value)
    with pytest.raises(ValueError):
        sd.sample_solution(ONE, max_trials=value)


@pytest.mark.parametrize("value", [0, 1, -0.1, float("nan")])
def test_bad_failure_probability(value):
    with pytest.raises(ValueError):
        sd.trial_budget(value)


@pytest.mark.parametrize("block", [1, 2, 3, 5])
def test_full_grover_circuit_matches_probability_and_keeps_work_clean(block):
    import numpy as np
    from qiskit.quantum_info import Statevector

    g = solved_grid(block)
    g[0][0] = 0
    size = sd.search_size(g)
    sv = Statevector.from_instruction(sd.grover_circuit(g, 2)).data
    assert abs(sv[0]) ** 2 == pytest.approx(grover_probability(size, 1, 2))
    np.testing.assert_allclose(sv[size:], 0, atol=1e-12)
