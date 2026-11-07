# Grover mini-project: general Sudoku

A clean reversible oracle and Grover sampler for n² × n² Sudoku with n × n
blocks. Includes executed 4×4 and 9×9 demonstrations with few blanks.

| File | Purpose |
|---|---|
| `src/grover_miniproject/sudoku.py` | Validation, oracle, search and resources |
| `src/grover_miniproject/theory.py` | Independent closed-form probability |
| `notebooks/sudoku.ipynb` | Design, drawings, demonstrations and budget proof |
| `tests/test_sudoku.py` | Automated correctness and regression checks |

## Setup and verification

Python 3.10+ and uv:

```bash
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/sudoku.ipynb
```

## Inputs and encoding

A nonempty square list of rows, side n² for any positive integer n; 0 means
blank and integer clues are 1 through n². Malformed shapes, noninteger clues
(including booleans), negative clues, and out-of-range clues raise `ValueError`
before circuit construction. Conflicting clues raise `ValueError` in oracle
builders; the sampler returns `None` without querying an oracle. Completed
consistent grids are returned directly by the sampler; circuit builders require
at least one blank.

For side s and k blanks, w = max(1, ceil(log2(s))) bits per blank encode digit−1
in little-endian order. The binary search size is N = 2**(w*k), including unused
codes: 9×9 uses 16**k, not 9**k. Domain flags permit only actual digits compatible
with clues, so invalid codes cannot be marked. Pair flags enforce inequality
between blanks in any shared row, column or block.

The oracle uses X, CX, CCX and Z only. Wider equality controls decompose into
Toffoli ladders with max(w−2, 0) shared clean scratch qubits. With V constraint
flags, another V−1 ancillas implement the final AND. Compute flags → AND and Z
→ undo AND → uncompute flags returns every work qubit to zero. The diffuser
uses one logical `mcx`, reducing to X when it has no controls.

## Guaranteed unknown-M search

Let $N=2^{wk}$ be the binary search size (including unused codes), and let
$M$ count valid completions. Each trial starts from the uniform state. Flip a fair
coin: use $j=0$ on one branch; on the other choose $j$ uniformly from
$0,\ldots,m-1$, with $m=\lceil2\sqrt N\rceil$.

For $1\le M\le N/2$, put $\theta=\arcsin\sqrt{M/N}$. Averaging the Grover
probability over $j$ gives
$$A_m=\frac12-\frac{\sin(4m\theta)}{4m\sin(2\theta)}.$$
Since $\sin(2\theta)=2\sqrt{(M/N)(1-M/N)}\ge1/\sqrt N$ and
$m\ge2\sqrt N$, $A_m\ge3/8\ge1/4$. Including the coin branch therefore gives
success at least $1/8$ per trial. If $M>N/2$, the zero-iteration branch alone
contributes more than $1/4$. Thus, for every $M>0$, $T$ independent trials fail
with probability at most $(7/8)^T$. Choosing
$$T=\left\lceil\frac{\log(1/3)}{\log(7/8)}\right\rceil=9$$
gives success at least $1-(7/8)^9\approx0.6993>2/3$. This is deliberately
conservative, does not require knowing $M$, and costs at most $T(m-1)$ oracle
calls. Each Grover iteration counts as **one** full compute/phase/uncompute
oracle invocation. Classical candidate validation is separate.

The guarantee assumes ideal noiseless operations and independent random draws;
a fixed seed is for reproducibility, not evidence of the probability guarantee.
Smaller explicit `max_trials` values have only the reported weaker bound. With
$M=0$, the algorithm returns `None` after its budget; `None` does not certify
unsatisfiability. Malformed inputs raise `ValueError`; conflicting clues return
`None` without any search. A consistent completed grid returns immediately.

The averaging identity is Lemma 2 of
[Boyer, Brassard, Høyer and Tapp, *Tight bounds on quantum searching*](https://arxiv.org/abs/quant-ph/9605034).
Our fixed-cap mixture uses that identity rather than claiming the original
adaptive BBHT stopping rule has a fixed-budget guarantee.

## Results and resources

`sample_solution(grid)` returns a validated solution or `None`.
`sample_solution_report(grid)` additionally returns actual trials, actual logical
oracle calls, the maximum call budget, and the failure bound if solvable.
The optional `failure_probability` sets the default trial budget;
`max_trials` explicitly overrides it, possibly weakening the guarantee.

`resources(grid)` reports bits per cell, binary search size, search/work qubits,
per-oracle gate counts/depth, default trials and the worst-case query budget.
Each full compute/phase/uncompute oracle counts once. Oracle gate counts exclude
diffusers and state preparation, and assume all-to-all connectivity. Multiply
per-oracle counts by actual calls to get the oracle portion of a run's cost.

The sampler is a classical search-register simulator using signs extracted from
the actual verified circuit. Its exhaustive sign preprocessing is O(N) circuit
basis evaluations, separate from the logical quantum oracle-call count. It is
not a fast quantum execution backend and does not demonstrate quantum speedup.
Use few blanks; full Statevector memory grows with all search and work qubits.

Tests cover representative sides 1, 4, 9, 16 and 25, unused codes, pair checks,
no-solution cases, phases, work cleanup, Qiskit complex superpositions, malformed
inputs, probability budgets and exact oracle-call accounting. Any positive
block side is accepted; simulation costs remain exponential in the blanks.
