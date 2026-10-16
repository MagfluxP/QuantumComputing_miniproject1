"""Closed-form Grover success probability, used as an independent reference."""

from __future__ import annotations

import math


def grover_probability(n_total: int, n_marked: int, iterations: int) -> float:
    """P(measuring a marked item) after `iterations` Grover rounds.

    Inputs: N = n_total items, M = n_marked solutions, uniform initial state.
    Output: sin^2((2k+1) * theta) with theta = asin(sqrt(M/N)).
    Assumption: ideal, noiseless oracle and diffuser. This is exact (not an
    approximation) for the two-dimensional rotation picture of Grover's algorithm.
    """
    if n_marked == 0:
        return 0.0
    theta = math.asin(math.sqrt(n_marked / n_total))
    return math.sin((2 * iterations + 1) * theta) ** 2


def optimal_iterations(n_total: int, n_marked: int) -> int:
    """Iteration count maximising success: round(pi / (4 theta) - 1/2)."""
    if n_marked == 0:
        return 0
    theta = math.asin(math.sqrt(n_marked / n_total))
    return max(0, round(math.pi / (4 * theta) - 0.5))
