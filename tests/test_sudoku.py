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
