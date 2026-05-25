"""
EntropyShield-X — Terminal Display Utilities
============================================
Pretty, readable output for educational demos (no external deps).
"""

from __future__ import annotations

import numpy as np


def banner(title: str, width: int = 70) -> None:
    """Print a major section header (ASCII-safe for Windows terminals)."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def subheader(title: str, width: int = 70) -> None:
    """Print a subsection header."""
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print(f"{'-' * width}")


def print_formula_block(lines: list[str]) -> None:
    """Print indented formula / explanation lines."""
    for line in lines:
        print(f"  {line}")


def print_distribution_table(
    labels: list[str],
    p: np.ndarray,
    q: np.ndarray,
    kl_value: float,
    unit: str = "nats",
) -> None:
    """Side-by-side P vs Q with per-outcome log-ratio insight."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = np.clip(q / q.sum(), 1e-15, 1.0)

    print(f"\n  {'Outcome':<16} {'P (true)':>10} {'Q (pred)':>10} {'P/Q':>10} {'P*log(P/Q)':>12}")
    print(f"  {'-' * 62}")

    total_check = 0.0
    for i, name in enumerate(labels):
        if p[i] > 0:
            ratio = p[i] / q[i]
            term = p[i] * np.log(ratio)
            total_check += term
            contrib = f"{term:+.4f}"
        else:
            ratio = float("nan")
            contrib = "  (skip)"

        print(
            f"  {name:<16} {p[i]:>9.2%} {q[i]:>9.2%} "
            f"{ratio if p[i] > 0 else 0:>10.4f} {contrib:>12}"
        )

    print(f"  {'-' * 62}")
    print(f"  sum P(x)*log(P(x)/Q(x)) =  {total_check:.6f}  ({unit})")
    print(f"  kl_divergence()        =  {kl_value:.6f}  ({unit})")


def verdict_kl(kl: float, low_threshold: float = 0.05, high_threshold: float = 0.5) -> str:
    """Human-readable interpretation of KL magnitude (natural log units)."""
    if kl < low_threshold:
        return "LOW divergence - Q closely matches P (good prediction / normal behavior)"
    if kl < high_threshold:
        return "MODERATE divergence - noticeable mismatch; review recommended"
    return "HIGH divergence - Q is very different from P (bad model or anomaly)"


def bar(probability: float, width: int = 24) -> str:
    """ASCII probability bar (# = mass, . = empty)."""
    filled = int(round(probability * width))
    return "#" * filled + "." * (width - filled)
