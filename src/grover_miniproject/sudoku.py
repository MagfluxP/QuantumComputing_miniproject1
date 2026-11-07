"""General n² × n² Sudoku with clean reversible Grover oracles.

Digits are encoded as digit - 1 in little-endian registers. Domain flags
reject unused binary codes as well as digits excluded by clues. Only X, CX,
CCX and Z occur in the oracle; wider controls use clean Toffoli ladders.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import product
from numbers import Integral

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

Grid = list[list[int]]


def grid_spec(grid: Grid) -> tuple[int, int, int]:
    """Validate shape and clue types/ranges; return (side, block side, bit width)."""
    if not isinstance(grid, (list, tuple)) or not grid:
        raise ValueError("grid must be a nonempty square sequence of rows")
    side = len(grid)
    block = math.isqrt(side)
    if block * block != side:
        raise ValueError("grid side must be a perfect square (n²)")
    if any(not isinstance(row, (list, tuple)) or len(row) != side for row in grid):
        raise ValueError("grid must have exactly side entries in every row")
    if any(
        isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= v <= side
        for row in grid
        for v in row
    ):
        raise ValueError(f"clues must be integers from 0 through {side}")
    return side, block, max(1, (side - 1).bit_length())


def is_valid_grid(g: Grid) -> bool:
    """Independent completed-grid checker; malformed grids return False."""
    try:
        side, block, _ = grid_spec(g)
    except ValueError:
        return False
    want = set(range(1, side + 1))
    return (
        all(set(row) == want for row in g)
        and all({g[r][c] for r in range(side)} == want for c in range(side))
        and all(
            {g[r][c] for r in range(br, br + block) for c in range(bc, bc + block)}
            == want
            for br in range(0, side, block)
            for bc in range(0, side, block)
        )
    )


def empties(grid: Grid) -> list[tuple[int, int]]:
    side, _, _ = grid_spec(grid)
    return [(r, c) for r in range(side) for c in range(side) if grid[r][c] == 0]


def fill(grid: Grid, values: tuple[int, ...]) -> Grid:
    out = [list(row) for row in grid]
    for (r, c), v in zip(empties(grid), values, strict=True):
        out[r][c] = v
    return out


def brute_force_solutions(grid: Grid) -> list[tuple[int, ...]]:
    """All completions (benchmark only), enumerating actual digits."""
    side, _, _ = grid_spec(grid)
    return [
        vals
        for vals in product(range(1, side + 1), repeat=len(empties(grid)))
        if is_valid_grid(fill(grid, vals))
    ]


def shares_unit(a: tuple[int, int], b: tuple[int, int], block: int = 2) -> bool:
    return (
        a[0] == b[0]
        or a[1] == b[1]
        or (a[0] // block, a[1] // block) == (b[0] // block, b[1] // block)
    )


def allowed_digits(grid: Grid, cell: tuple[int, int]) -> list[int]:
    """Valid digits not excluded by clues; unused binary codes never qualify."""
    side, block, _ = grid_spec(grid)
    used = {
        grid[r][c]
        for r in range(side)
        for c in range(side)
        if grid[r][c] and (r, c) != cell and shares_unit(cell, (r, c), block)
    }
    return [d for d in range(1, side + 1) if d not in used]


def clues_consistent(grid: Grid) -> bool:
    side, block, _ = grid_spec(grid)
    cl = [(r, c) for r in range(side) for c in range(side) if grid[r][c]]
    return all(
        not (grid[a[0]][a[1]] == grid[b[0]][b[1]] and shares_unit(a, b, block))
        for i, a in enumerate(cl)
        for b in cl[i + 1 :]
    )


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


def pairs(grid: Grid) -> list[tuple[int, int]]:
    """Index pairs (i, j) of empty cells that share a row, column or block."""
    cells = empties(grid)
    _, block, _ = grid_spec(grid)
    return [
        (i, j)
        for i in range(len(cells))
        for j in range(i + 1, len(cells))
        if shares_unit(cells[i], cells[j], block)
    ]


def layout(grid: Grid) -> tuple[list[int], list[int], list[int]]:
    _, _, width = grid_spec(grid)
    k = len(empties(grid))
    v = k + len(pairs(grid))
    search = list(range(width * k))
    flags = list(range(width * k, width * k + v))
    # AND ladder first, then shared scratch for wide equality checks.
    scratch = max(width - 2, 0) if k else 0
    anc = list(range(width * k + v, width * k + v + max(v - 1, 0) + scratch))
    return search, flags, anc


# --------------------------------------------------------------------------- #
# Oracle: gate list so the same block can be applied forward and reversed
# (every gate used here -- X, CX, CCX -- is its own inverse)
# --------------------------------------------------------------------------- #


def _controlled_x_ops(
    controls: list[int], target: int, scratch: list[int]
) -> list[tuple]:
    """Clean multi-control X, decomposed exclusively into X/CX/CCX."""
    if not controls:
        return [("x", target)]
    if len(controls) == 1:
        return [("cx", controls[0], target)]
    if len(controls) == 2:
        return [("ccx", *controls, target)]
    ladder = [("ccx", controls[0], controls[1], scratch[0])]
    ladder += [
        ("ccx", controls[i], scratch[i - 2], scratch[i - 1])
        for i in range(2, len(controls) - 1)
    ]
    return (
        ladder
        + [("ccx", controls[-1], scratch[len(controls) - 3], target)]
        + list(reversed(ladder))
    )


def _eq_const_ops(
    qubits: list[int], code: int, target: int, scratch: list[int]
) -> list[tuple]:
    zeros = [q for i, q in enumerate(qubits) if not (code >> i) & 1]
    return (
        [("x", q) for q in zeros]
        + _controlled_x_ops(qubits, target, scratch)
        + [("x", q) for q in zeros]
    )


def _compute_ops(grid: Grid) -> list[tuple]:
    search, flags, anc = layout(grid)
    _, _, width = grid_spec(grid)
    scratch = anc[max(len(flags) - 1, 0) :]
    cells = empties(grid)
    ops: list[tuple] = []
    for i, cell in enumerate(cells):
        qubits = search[width * i : width * (i + 1)]
        for d in allowed_digits(grid, cell):
            ops += _eq_const_ops(qubits, d - 1, flags[i], scratch)
    for p, (i, j) in enumerate(pairs(grid)):
        a = search[width * i : width * (i + 1)]
        b = search[width * j : width * (j + 1)]
        f = flags[len(cells) + p]
        compare = [("cx", qa, qb) for qa, qb in zip(a, b, strict=True)]
        compare += [("x", qb) for qb in b]
        ops += compare + _controlled_x_ops(b, f, scratch)
        ops += list(reversed(compare)) + [("x", f)]
    return ops


def _apply(qc: QuantumCircuit, ops: list[tuple], reverse: bool = False) -> None:
    for op in reversed(ops) if reverse else ops:
        getattr(qc, op[0])(*op[1:])


def _validate(grid: Grid) -> None:
    grid_spec(grid)
    if not clues_consistent(grid):
        raise ValueError("clues conflict: no valid completion exists")
    if not empties(grid):
        raise ValueError("grid has no empty cells")


def _new_circuit(grid: Grid, name: str) -> QuantumCircuit:
    search, flags, anc = layout(grid)
    return QuantumCircuit(len(search) + len(flags) + len(anc), name=name)


def compute_checks_circuit(grid: Grid) -> QuantumCircuit:
    """Block 1: write every constraint result into its flag qubit."""
    _validate(grid)
    qc = _new_circuit(grid, "compute_checks")
    _apply(qc, _compute_ops(grid))
    return qc


def phase_circuit(grid: Grid) -> QuantumCircuit:
    """Block 2: AND all flags with a Toffoli ladder, Z on the result, undo ladder."""
    _validate(grid)
    _, flags, anc = layout(grid)
    qc = _new_circuit(grid, "and_phase")
    v = len(flags)
    if v == 1:
        qc.z(flags[0])
    else:
        ladder = [("ccx", flags[0], flags[1], anc[0])]
        ladder += [("ccx", flags[i], anc[i - 2], anc[i - 1]) for i in range(2, v)]
        _apply(qc, ladder)
        qc.z(anc[v - 2])
        _apply(qc, ladder, reverse=True)
    return qc


def uncompute_checks_circuit(grid: Grid) -> QuantumCircuit:
    """Block 3: exact inverse of block 1 (all gates are self-inverse, so reverse)."""
    _validate(grid)
    qc = _new_circuit(grid, "uncompute_checks")
    _apply(qc, _compute_ops(grid), reverse=True)
    return qc


def sudoku_oracle(grid: Grid) -> QuantumCircuit:
    """S_f|z>|0> = (-1)^f(z)|z>|0>, f(z)=1 iff the filled grid is a valid Sudoku."""
    _validate(grid)
    qc = _new_circuit(grid, "oracle")
    qc.compose(compute_checks_circuit(grid), inplace=True)
    qc.compose(phase_circuit(grid), inplace=True)
    qc.compose(uncompute_checks_circuit(grid), inplace=True)
    return qc


def grover_block_circuit(grid: Grid) -> QuantumCircuit:
    """One Grover iteration drawn as labelled blocks (for qc.draw('mpl'))."""
    _validate(grid)
    search, flags, anc = layout(grid)
    n_q = len(search) + len(flags) + len(anc)
    qc = QuantumCircuit(n_q, name="grover_blocks")
    qc.h(search)
    blocks = [
        (compute_checks_circuit(grid), "Compute checks"),
        (phase_circuit(grid), "AND + phase flip"),
        (uncompute_checks_circuit(grid), "Uncompute checks"),
        (diffuser(grid), "Diffuser"),
    ]
    for sub, label in blocks:
        qc.append(sub.to_gate(label=label), list(range(n_q)))
    return qc


def diffuser(grid: Grid) -> QuantumCircuit:
    """Inversion about the uniform state on the search register (uses one MCX)."""
    _validate(grid)
    search, flags, anc = layout(grid)
    qc = QuantumCircuit(len(search) + len(flags) + len(anc), name="diffuser")
    qc.h(search)
    qc.x(search)
    qc.h(search[-1])
    qc.mcx(search[:-1], search[-1])
    qc.h(search[-1])
    qc.x(search)
    qc.h(search)
    return qc


def grover_circuit(grid: Grid, iterations: int) -> QuantumCircuit:
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, Integral)
        or iterations < 0
    ):
        raise ValueError("iterations must be a nonnegative integer")
    search, _, _ = layout(grid)
    oracle, diff = sudoku_oracle(grid), diffuser(grid)
    qc = QuantumCircuit(oracle.num_qubits)
    qc.h(search)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diff, inplace=True)
    return qc


# --------------------------------------------------------------------------- #
# Verification and experiments
# --------------------------------------------------------------------------- #


def decode(z: int, k: int, width: int = 2) -> tuple[int, ...]:
    """Decode digit - 1; unused codes deliberately yield out-of-range digits."""
    return tuple(((z >> (width * i)) & ((1 << width) - 1)) + 1 for i in range(k))


def search_size(grid: Grid) -> int:
    """Binary candidate count, including unused codes (e.g. 16**k for 9×9)."""
    return 1 << len(layout(grid)[0])


def simulate_basis(qc: QuantumCircuit, bits: int) -> tuple[int, int]:
    """Run a circuit made only of X, CX, CCX, Z on a computational basis state.

    Returns (output bit string, sign). These gates map basis states to basis
    states (up to the -1 from Z), so this is an exact, instant classical
    simulation of the oracle -- no statevector needed. Bit q of `bits` = qubit q.
    """
    sign = 1
    for inst in qc.data:
        name = inst.operation.name
        q = [qc.find_bit(b).index for b in inst.qubits]
        if name == "x":
            bits ^= 1 << q[0]
        elif name == "cx":
            if (bits >> q[0]) & 1:
                bits ^= 1 << q[1]
        elif name == "ccx":
            if (bits >> q[0]) & 1 and (bits >> q[1]) & 1:
                bits ^= 1 << q[2]
        elif name == "z":
            if (bits >> q[0]) & 1:
                sign = -sign
        else:
            raise ValueError(f"unsupported gate {name}")
    return bits, sign


def oracle_signs(grid: Grid) -> list[int]:
    """Sign (-1)^f(z) the oracle applies to |z>|0> for every z, computed from the
    circuit itself. Asserts every work qubit returns to |0>."""
    oracle = sudoku_oracle(grid)
    signs = []
    for z in range(search_size(grid)):
        out, sign = simulate_basis(oracle, z)  # work qubits start at 0
        assert out == z, f"work qubits not restored for z={z}"
        signs.append(sign)
    return signs


def check_oracle(grid: Grid) -> None:
    """Oracle sign equals the independent classical checker for ALL assignments,
    and all work qubits are uncomputed. The oracle is a reversible permutation plus
    a Z, so correct basis-state action implies correct action on superpositions
    (linearity). Exact: no tolerance needed."""
    k = len(empties(grid))
    for z, sign in enumerate(oracle_signs(grid)):
        ok = is_valid_grid(fill(grid, decode(z, k, grid_spec(grid)[2])))
        assert sign == (-1 if ok else 1), (z, ok)


def check_oracle_statevector(grid: Grid, atol: float = 1e-12) -> None:
    """Same check with Qiskit's Statevector. Only for tiny puzzles (<= ~14 qubits)."""
    k = len(empties(grid))
    oracle = sudoku_oracle(grid)
    for z in range(search_size(grid)):
        ok = is_valid_grid(fill(grid, decode(z, k, grid_spec(grid)[2])))
        sv = Statevector.from_int(z, 2**oracle.num_qubits).evolve(oracle)
        assert abs(sv.data[z] - (-1.0 if ok else 1.0)) < atol, (z, ok)


def _grover_states(grid: Grid, max_iter: int):
    """Yield the search-register amplitudes after 0..max_iter Grover iterations.

    Valid because the oracle is diagonal on the search register when the work
    qubits are clean (verified by oracle_signs): oracle = sign flip using the
    circuit's own signs, diffuser = 2|s><s| - I (equal to the circuit's diffuser up
    to a global phase). This avoids simulating all work qubits.
    """
    signs = np.array(oracle_signs(grid), dtype=float)
    dim = len(signs)
    amp = np.full(dim, 1 / math.sqrt(dim))
    for _ in range(max_iter + 1):
        yield amp
        amp = signs * amp
        amp = 2 * amp.mean() - amp


def success_curve(grid: Grid, max_iter: int) -> list[float]:
    """P(valid completion) after 0..max_iter Grover iterations (exact)."""
    good = np.array(oracle_signs(grid)) == -1
    return [float((a[good] ** 2).sum()) for a in _grover_states(grid, max_iter)]


def trial_budget(failure_probability: float = 1 / 3) -> int:
    """Each independent trial succeeds with probability >= 1/8 if M > 0."""
    if not 0 < failure_probability < 1:
        raise ValueError("failure_probability must be between 0 and 1")
    return math.ceil(math.log(failure_probability) / math.log(7 / 8))


@dataclass(frozen=True)
class SearchResult:
    solution: Grid | None
    trials: int
    oracle_calls: int
    oracle_call_budget: int
    failure_bound_if_solvable: float


def sample_solution_report(
    grid: Grid,
    max_trials: int | None = None,
    seed: int | None = None,
    failure_probability: float = 1 / 3,
) -> SearchResult:
    """Unknown-M search with a proven budget; see README for the bound.

    Per trial: with probability 1/2 use zero iterations, otherwise choose j
    uniformly from 0..ceil(2 sqrt(N))-1. Counts are logical quantum oracle
    queries; classical sign-table preprocessing is separate simulation overhead.
    Explicit smaller budgets have only the reported weaker guarantee.
    """
    grid_spec(grid)  # malformed input is an error, not a search failure
    required = trial_budget(failure_probability)
    budget = required if max_trials is None else max_trials
    if isinstance(budget, bool) or not isinstance(budget, Integral) or budget < 0:
        raise ValueError("max_trials must be a nonnegative integer")
    if not clues_consistent(grid):
        return SearchResult(None, 0, 0, 0, 0.0)
    if not empties(grid):
        return SearchResult([list(row) for row in grid], 0, 0, 0, 0.0)
    size = search_size(grid)
    cap = math.isqrt(4 * size - 1) + 1  # ceil(2 sqrt(N)), using integer arithmetic
    bound = (7 / 8) ** budget
    call_budget = budget * (cap - 1)
    if not budget:
        return SearchResult(None, 0, 0, call_budget, bound)
    signs = np.asarray(oracle_signs(grid), dtype=float)
    rng = random.Random(seed)
    calls = 0
    for trial in range(1, budget + 1):
        reps = 0 if rng.randrange(2) == 0 else rng.randrange(cap)
        amp = np.full(size, 1 / math.sqrt(size))
        for _ in range(reps):
            amp *= signs
            amp = 2 * amp.mean() - amp
        calls += reps
        z = rng.choices(range(size), weights=amp**2)[0]
        candidate = fill(grid, decode(z, len(empties(grid)), grid_spec(grid)[2]))
        if is_valid_grid(candidate):
            return SearchResult(candidate, trial, calls, call_budget, bound)
    return SearchResult(None, budget, calls, call_budget, bound)


def sample_solution(
    grid: Grid,
    max_trials: int | None = None,
    seed: int | None = None,
    failure_probability: float = 1 / 3,
) -> Grid | None:
    """Return a validated solution or None; default success is >= 2/3 if solvable."""
    return sample_solution_report(grid, max_trials, seed, failure_probability).solution


def resources(grid: Grid) -> dict:
    """Qubit counts, oracle gate counts (already in {x,cx,ccx,z}) and depth."""
    o = sudoku_oracle(grid)
    s, f, a = layout(grid)
    size = search_size(grid)
    trials = trial_budget()
    cap = math.isqrt(4 * size - 1) + 1
    return {
        "grid_side": len(grid),
        "bits_per_cell": grid_spec(grid)[2],
        "search_size": size,
        "default_trials": trials,
        "oracle_call_budget": trials * (cap - 1),
        "search_qubits": len(s),
        "work_qubits": len(f) + len(a),
        "oracle_gates": dict(o.count_ops()),
        "oracle_depth": o.depth(),
    }
