# Grover mini-project: 4x4 Sudoku

A Grover search circuit for completions of a partially filled 4x4 Sudoku, built
from H, X, Z, CX and Toffoli gates. The only multi-controlled gate is one `mcx`
inside the diffuser.

| Part | File |
|---|---|
| Circuit, oracle, sampler | `src/grover_miniproject/sudoku.py` |
| Closed-form Grover probability (test reference) | `src/grover_miniproject/theory.py` |
| Explanation, circuit drawings, experiments | `notebooks/sudoku.ipynb` |
| Tests | `tests/test_sudoku.py` |

## Setup (Python 3.10+, managed with uv)

```bash
uv sync                     # creates .venv and writes uv.lock (first time only)
uv sync --locked            # reproduce the exact locked environment afterwards
uv run pytest               # run the tests
uv run jupyter lab          # open notebooks/sudoku.ipynb
```

Run `uv sync` once to create `uv.lock`, then commit `pyproject.toml` and
`uv.lock`. Before submitting: restart the kernel and **Run All** so the notebook
saves its outputs.

Pre-submission checks:

```bash
uv sync --locked
uv run pytest
uv run ruff format .        # one-time auto-format (also formats the notebook)
uv run ruff check . --fix
uv run ruff check .
uv run ruff format --check .
```

## Design in brief

* Each blank cell gets 2 search qubits holding digit - 1 (little-endian, qubit 0 =
  least significant bit). Clues are classical constants and take no qubits.
* Flag qubits (1 = satisfied): one "domain" flag per blank (digit not excluded by a
  clue in its row, column or block) and one "differ" flag per pair of blanks that
  share a unit. V flags need V - 1 ancillas for a Toffoli AND ladder.
* Oracle = compute flags, AND them and apply a single-qubit Z, uncompute the flags.
  Work qubits start and end in |0>.
* Diffuser = H X (phase flip on all-ones) X H with one `mcx`.
* No measurements or classical bits inside any circuit. Sampling (`sample_solution`)
  uses a BBHT random iteration count, validates every result with an independent
  classical checker, and returns `None` ("no solution found within budget") when
  the trial budget runs out.

## Supported inputs

4x4 grids as lists of lists, 0 = empty, 1-4 = clue. Use few blank cells (the
number of work qubits grows quickly with the number of blanks). Conflicting clues
raise `ValueError` in the oracle builders and make the sampler return `None`.

## Known limitations

* Only 4x4 grids (n = 2). Larger grids need wider digit registers and a range check.
* Full-circuit statevector simulation is limited to about 20 qubits. Success curves
  and sampling therefore simulate only the search register using the oracle's
  verified signs (valid because the oracle is diagonal there when the work qubits
  are clean). Circuit resources are reported separately from simulation time.
* Gate counts assume all-to-all connectivity and the gates the oracle uses (X, CX,
  Toffoli, Z).
