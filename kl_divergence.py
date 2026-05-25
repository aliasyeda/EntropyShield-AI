"""
EntropyShield-X — KL Divergence Core
====================================

WHAT IS KL DIVERGENCE? (Kullback–Leibler divergence)
----------------------------------------------------
KL divergence answers one question:

    "If reality follows distribution P, how much extra surprise do I get
     by betting with distribution Q instead of P?"

Formula (discrete case):

    D_KL(P || Q) = Σ  P(x) · log( P(x) / Q(x) )

    Equivalently:  D_KL(P || Q) = Σ  P(x) · ( log P(x) - log Q(x) )

Where:
  - P(x) = TRUE probability of outcome x  (ground truth, "reality")
  - Q(x) = PREDICTED probability of outcome x  (model guess)
  - log  = natural logarithm (ln) in ML; log₂ in information theory

KEY PROPERTIES (explained in comments throughout this file)
-----------------------------------------------------------
1. NOT SYMMETRIC
   D_KL(P || Q) ≠ D_KL(Q || P) in general.
   "How wrong is Q when truth is P?" is different from the reverse question.

2. ZERO WHEN IDENTICAL
   If P == Q everywhere, then P(x)/Q(x) = 1 and log(1) = 0, so KL = 0.

3. NON-NEGATIVE
   KL ≥ 0 always (Gibbs' inequality). Larger KL → predictions differ more from truth.

4. NOT A TRUE "DISTANCE"
   It does not satisfy the triangle inequality, so it is a "divergence," not a metric.

RELATIONSHIP: ENTROPY, CROSS-ENTROPY, KL DIVERGENCE
----------------------------------------------------
  Entropy:         H(P)           = -Σ P(x) log P(x)
  Cross-entropy:   H(P, Q)        = -Σ P(x) log Q(x)
  KL divergence:   D_KL(P || Q)   =  Σ P(x) log( P(x) / Q(x) )

Fundamental identity:

    Cross-Entropy = Entropy + KL Divergence
    H(P, Q) = H(P) + D_KL(P || Q)

So KL measures the *extra* cost of using Q instead of the perfect code P.
When Q = P, cross-entropy is as low as possible (equals entropy).

AI / TRANSFORMER / LLM CONNECTION
---------------------------------
  - Language models output Q (next-token probabilities); training data defines P.
  - Low KL between model Q and empirical P → model matches real language patterns.
  - Fine-tuning minimizes KL (or cross-entropy, equivalent up to constant H(P)).

CYBERSECURITY / ANOMALY DETECTION
---------------------------------
  - P = baseline "normal" traffic profile; Q = live observed profile.
  - Spike in KL → behavior drift, possible attack or exfiltration.
  - Adversarial inputs can push Q far from P while looking superficially normal.
"""

from __future__ import annotations

import numpy as np

from cross_entropy import cross_entropy
from entropy import shannon_entropy


# Tiny floor so log(Q(x)) never hits -infinity when Q(x) ≈ 0
_EPS = 1e-15


def kl_divergence(
    true_probs: np.ndarray,
    predicted_probs: np.ndarray,
    base: str = "natural",
) -> float:
    """
    Compute Kullback–Leibler divergence D_KL(P || Q).

    Parameters
    ----------
    true_probs : array-like
        Ground-truth distribution P (what actually happens).
    predicted_probs : array-like
        Model / observed distribution Q (what we predict or measure).
    base : str
        "natural" (ln, standard in deep learning) or "2" (bits).

    Returns
    -------
    float
        KL divergence (0 when P == Q; larger = more different).

    Notes
    -----
    - Only outcomes with P(x) > 0 contribute (0 · log(...) = 0).
    - If Q(x) = 0 where P(x) > 0, KL is theoretically +∞; we clip Q for demos.
    """
    # Step 1: Convert inputs to NumPy arrays (64-bit float for stability)
    p = np.asarray(true_probs, dtype=np.float64)
    q = np.asarray(predicted_probs, dtype=np.float64)

    if p.shape != q.shape:
        raise ValueError("true_probs and predicted_probs must have the same length.")

    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Probabilities must be non-negative.")

    # Step 2: Normalize so each distribution sums to 1.0
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum == 0 or q_sum == 0:
        raise ValueError("Probability sums must be positive.")
    p = p / p_sum
    q = q / q_sum

    # Step 3: Clip Q away from zero (avoids log(0) in educational demos)
    q = np.clip(q, _EPS, 1.0)

    # Step 4: Pick logarithm — ln for ML, log₂ for bits
    if base == "2":
        log_fn = np.log2
    else:
        log_fn = np.log

    # Step 5: Core formula  D_KL(P||Q) = Σ P(x) · log( P(x) / Q(x) )
    # We only sum over x where P(x) > 0 (zero true mass → zero contribution)
    mask = p > 0
    p_nz = p[mask]
    q_nz = q[mask]

    # log(P/Q) = log(P) - log(Q)  — numerically stable when computed this way
    kl = np.sum(p_nz * (log_fn(p_nz) - log_fn(q_nz)))

    return float(kl)


def verify_cross_entropy_identity(
    true_probs: np.ndarray,
    predicted_probs: np.ndarray,
) -> dict[str, float]:
    """
    Demonstrate: Cross-Entropy = Entropy + KL Divergence.

    Returns H(P), H(P,Q), D_KL(P||Q), and the residual |CE - H - KL|
    (should be ~0 up to floating-point error).
    """
    p = np.asarray(true_probs, dtype=np.float64)
    q = np.asarray(predicted_probs, dtype=np.float64)

    h_p = shannon_entropy(p, base=np.e)  # nats (natural units)
    h_pq = cross_entropy(p, q, base="natural")
    kl = kl_divergence(p, q, base="natural")

    residual = abs(h_pq - (h_p + kl))

    return {
        "entropy_H_P": h_p,
        "cross_entropy_H_PQ": h_pq,
        "kl_D_KL": kl,
        "sum_H_plus_KL": h_p + kl,
        "residual": residual,
    }


def kl_symmetry_gap(
    true_probs: np.ndarray,
    predicted_probs: np.ndarray,
) -> tuple[float, float, float]:
    """
    Show KL is NOT symmetric: return (KL(P||Q), KL(Q||P), absolute difference).
    """
    kl_pq = kl_divergence(true_probs, predicted_probs)
    kl_qp = kl_divergence(predicted_probs, true_probs)
    return kl_pq, kl_qp, abs(kl_pq - kl_qp)
