"""4x4 Sudoku (n=2) with Grover's algorithm in Qiskit.

Grid input: 4x4 list of lists, 0 = empty, 1..4 = clue.

Encoding: each empty cell gets 2 search qubits holding digit-1 in binary
(little-endian: first qubit = least significant bit). Empty cells are ordered
row-major. Clues are classical and never stored in qubits.

Register layout (qubit indices):
    search : 2k qubits, cell i uses qubits (2i, 2i+1)
    flags  : V qubits, 1 = "this constraint is satisfied"
             - one "domain" flag per empty cell (digit is not excluded by any clue
               in its row/column/block)
             - one "differ" flag per pair of empty cells that share a unit
    anc    : max(V-1, 0) ancillas for the Toffoli AND ladder
Flags and ancillas start and end in |0>.

Oracle: compute flags -> AND them with a Toffoli ladder -> Z on the result
(single-qubit gate, no multi-controlled Z) -> uncompute. The only multi-controlled
gate is in the diffuser, as the project allows.
"""

from __future__ import annotations

import math
import random
from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

Grid = list[list[int]]
N = 4  # grid side (n = 2)

# --------------------------------------------------------------------------- #
# Classical helpers (independent checker, used only for verification/validation)
# --------------------------------------------------------------------------- #


def is_valid_grid(g: Grid) -> bool:
    """Every row, column and 2x2 block contains 1..4 exactly once."""
    want = {1, 2, 3, 4}
    rows = all(set(r) == want for r in g)
    cols = all({g[r][c] for r in range(N)} == want for c in range(N))
    blocks = all(
        {g[r][c] for r in range(br, br + 2) for c in range(bc, bc + 2)} == want
        for br in (0, 2)
        for bc in (0, 2)
    )
    return rows and cols and blocks


def empties(grid: Grid) -> list[tuple[int, int]]:
    return [(r, c) for r in range(N) for c in range(N) if grid[r][c] == 0]


def fill(grid: Grid, values: tuple[int, ...]) -> Grid:
    out = [row[:] for row in grid]
    for (r, c), v in zip(empties(grid), values, strict=True):
        out[r][c] = v
    return out


def brute_force_solutions(grid: Grid) -> list[tuple[int, ...]]:
    """All completions (benchmark only). Values are digits 1..4."""
    k = len(empties(grid))
    return [
        vals
        for vals in product(range(1, 5), repeat=k)
        if is_valid_grid(fill(grid, vals))
    ]


def shares_unit(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return (
        a[0] == b[0] or a[1] == b[1] or (a[0] // 2, a[1] // 2) == (b[0] // 2, b[1] // 2)
    )


def allowed_digits(grid: Grid, cell: tuple[int, int]) -> list[int]:
    """Digits not excluded by clues in the cell's row/column/block."""
    used = {
        grid[r][c]
        for r in range(N)
        for c in range(N)
        if grid[r][c] and (r, c) != cell and shares_unit(cell, (r, c))
    }
    return [d for d in range(1, 5) if d not in used]


def clues_consistent(grid: Grid) -> bool:
    """No two equal clues share a unit."""
    cl = [(r, c) for r in range(N) for c in range(N) if grid[r][c]]
    return all(
        not (grid[a[0]][a[1]] == grid[b[0]][b[1]] and shares_unit(a, b))
        for i, a in enumerate(cl)
        for b in cl[i + 1 :]
    )


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


def pairs(grid: Grid) -> list[tuple[int, int]]:
    """Index pairs (i, j) of empty cells that share a row, column or block."""
    cells = empties(grid)
    return [
        (i, j)
        for i in range(len(cells))
        for j in range(i + 1, len(cells))
        if shares_unit(cells[i], cells[j])
    ]


def layout(grid: Grid) -> tuple[list[int], list[int], list[int]]:
    k = len(empties(grid))
    v = k + len(pairs(grid))
    search = list(range(2 * k))
    flags = list(range(2 * k, 2 * k + v))
    anc = list(range(2 * k + v, 2 * k + v + max(v - 1, 0)))
    return search, flags, anc


# --------------------------------------------------------------------------- #
# Oracle: gate list so the same block can be applied forward and reversed
# (every gate used here -- X, CX, CCX -- is its own inverse)
# --------------------------------------------------------------------------- #


def _eq_const_ops(q0: int, q1: int, code: int, target: int) -> list[tuple]:
    """target ^= [cell == code]; cell qubits (q0 LSB, q1) restored."""
    zeros = [q for q, bit in ((q0, code & 1), (q1, code >> 1)) if bit == 0]
    return (
        [("x", q) for q in zeros]
        + [("ccx", q0, q1, target)]
        + [("x", q) for q in zeros]
    )


def _compute_ops(grid: Grid) -> list[tuple]:
    search, flags, _ = layout(grid)
    cells = empties(grid)
    ops: list[tuple] = []
    # domain flags: flag = OR over allowed digits of [cell == digit] (mutually exclusive)
    for i, cell in enumerate(cells):
        q0, q1 = search[2 * i], search[2 * i + 1]
        for d in allowed_digits(grid, cell):
            ops += _eq_const_ops(q0, q1, d - 1, flags[i])
    # pair flags: flag = [cell_i != cell_j]
    for p, (i, j) in enumerate(pairs(grid)):
        a0, a1 = search[2 * i], search[2 * i + 1]
        b0, b1 = search[2 * j], search[2 * j + 1]
        f = flags[len(cells) + p]
        ops += [
            ("cx", a0, b0),
            ("cx", a1, b1),
            ("x", b0),
            ("x", b1),
        ]  # b_k = (a_k == b_k)
        ops += [("ccx", b0, b1, f)]  # f = [equal]
        ops += [("x", b0), ("x", b1), ("cx", a1, b1), ("cx", a0, b0)]  # restore b
        ops += [("x", f)]  # f = [different]
    return ops


def _apply(qc: QuantumCircuit, ops: list[tuple], reverse: bool = False) -> None:
    for op in reversed(ops) if reverse else ops:
        getattr(qc, op[0])(*op[1:])


def _validate(grid: Grid) -> None:
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
    qc = _new_circuit(grid, "oracle")
    qc.compose(compute_checks_circuit(grid), inplace=True)
    qc.compose(phase_circuit(grid), inplace=True)
    qc.compose(uncompute_checks_circuit(grid), inplace=True)
    return qc


def grover_block_circuit(grid: Grid) -> QuantumCircuit:
    """One Grover iteration drawn as labelled blocks (for qc.draw('mpl'))."""
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


def decode(z: int, k: int) -> tuple[int, ...]:
    """Integer z (bit 2i,2i+1 = cell i) -> digits 1..4."""
    return tuple(((z >> (2 * i)) & 3) + 1 for i in range(k))


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
    k = len(empties(grid))
    oracle = sudoku_oracle(grid)
    signs = []
    for z in range(4**k):
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
        ok = is_valid_grid(fill(grid, decode(z, k)))
        assert sign == (-1 if ok else 1), (z, ok)


def check_oracle_statevector(grid: Grid, atol: float = 1e-12) -> None:
    """Same check with Qiskit's Statevector. Only for tiny puzzles (<= ~14 qubits)."""
    k = len(empties(grid))
    oracle = sudoku_oracle(grid)
    for z in range(4**k):
        ok = is_valid_grid(fill(grid, decode(z, k)))
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


def sample_solution(
    grid: Grid, max_trials: int = 20, seed: int | None = 0, growth: float = 6 / 5
) -> Grid | None:
    """Return a validated completed grid, or None ("no solution found within budget").

    Unknown number of solutions: random iteration count k < cap, cap grows by
    `growth` each failed trial (BBHT). Output is always classically validated, so
    there are no false positives.
    """
    if not clues_consistent(grid):
        return None
    k_cells = len(empties(grid))
    rng = random.Random(seed)
    cap, limit = 1.0, math.sqrt(4**k_cells)
    for _ in range(max_trials):
        reps = rng.randrange(max(1, math.ceil(cap)))
        amp = list(_grover_states(grid, reps))[-1]
        z = rng.choices(range(4**k_cells), weights=amp**2)[0]
        candidate = fill(grid, decode(z, k_cells))
        if is_valid_grid(candidate):
            return candidate
        cap = min(cap * growth, limit)
    return None


def resources(grid: Grid) -> dict:
    """Qubit counts, oracle gate counts (already in {x,cx,ccx,z}) and depth."""
    s, f, a = layout(grid)
    o = sudoku_oracle(grid)
    return {
        "search_qubits": len(s),
        "work_qubits": len(f) + len(a),
        "oracle_gates": dict(o.count_ops()),
        "oracle_depth": o.depth(),
    }
