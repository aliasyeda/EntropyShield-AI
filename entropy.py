"""
EntropyShield-X — Shannon Entropy Calculator
=============================================

WHAT IS ENTROPY?
----------------
Entropy measures *uncertainty* or *surprise* in a system.

Imagine you flip a coin:
  - A fair coin (50% heads, 50% tails) → HIGH entropy (very uncertain outcome)
  - A biased coin (99% heads, 1% tails) → LOW entropy (outcome is almost predictable)

In information theory, Shannon entropy (H) tells us the average number of "bits"
of surprise we get when we observe one outcome from a probability distribution.

Formula:  H = - Σ p(x) · log₂(p(x))

Where:
  - p(x) = probability of outcome x
  - log₂  = logarithm base 2 (results are in "bits" of information)
  - The minus sign makes the result positive (log of probabilities is negative)

UNCERTAINTY IN AI SYSTEMS
-------------------------
Modern AI models (classifiers, language models, anomaly detectors) output
*probability distributions* over possible outcomes. Entropy quantifies how
confident or uncertain those predictions are:

  - LOW entropy  → model is confident (one outcome dominates)
  - HIGH entropy → model is uncertain (many outcomes seem equally likely)

Examples:
  - A spam filter that says "99% spam" has low entropy → confident decision
  - A medical AI that says "40% disease A, 35% disease B, 25% healthy"
    has high entropy → uncertain, may need human review

ENTROPY IN AI SECURITY & ANOMALY DETECTION
------------------------------------------
Security systems use entropy to spot unusual behavior:

1. MODEL CONFIDENCE MONITORING
   Sudden spikes in prediction entropy can signal:
   - Adversarial inputs (crafted to confuse the model)
   - Distribution shift (data the model wasn't trained on)
   - Model degradation or poisoning

2. ANOMALY DETECTION
   Normal traffic/behavior often has predictable patterns (low entropy).
   Attacks or anomalies introduce randomness or rare patterns (high entropy):
   - Network packet timing with unusual variance
   - Login attempts from many IPs (high entropy in source-IP patterns)
   - Malware that encrypts data (high entropy in file contents)

3. DATA EXFILTRATION DETECTION
   Encrypted or compressed data has very high byte-level entropy.
   Security tools flag files/streams whose entropy exceeds a threshold.

4. PROMPT INJECTION / JAILBREAK DETECTION
   Some defenses measure entropy of token distributions or embedding
   activations to detect adversarial prompts that push the model off-manifold.
"""

import numpy as np


def shannon_entropy(probabilities: np.ndarray, base: float = 2) -> float:
    """
    Calculate Shannon entropy for a discrete probability distribution.

    Parameters
    ----------
    probabilities : array-like
        Probabilities for each outcome. Must sum to ~1.0 and be non-negative.
    base : float
        Logarithm base (2 = bits, e = nats, 10 = bans/hartleys). Default is 2.

    Returns
    -------
    float
        Entropy value in the chosen unit (bits if base=2).
    """
    # Convert input to a NumPy array of 64-bit floats for numerical stability
    p = np.asarray(probabilities, dtype=np.float64)

    # Reject invalid inputs early — probabilities cannot be negative
    if np.any(p < 0):
        raise ValueError("Probabilities must be non-negative.")

    # Normalize so probabilities sum to 1.0 (handles unnormalized inputs gracefully)
    total = p.sum()
    if total == 0:
        raise ValueError("Probabilities must sum to a positive value.")
    p = p / total

    # Only keep outcomes with p > 0 (log(0) is undefined; skip them)
    # Outcomes with zero probability contribute nothing to entropy
    p_nonzero = p[p > 0]

    # Core formula: H = - Σ p(x) · log(p(x))
    # np.log(p) / np.log(base) converts natural log to log with any base
    entropy = -np.sum(p_nonzero * (np.log(p_nonzero) / np.log(base)))

    return float(entropy)


def entropy_from_counts(counts: np.ndarray, base: float = 2) -> float:
    """
    Calculate entropy directly from occurrence counts (e.g., word frequencies).

    Parameters
    ----------
    counts : array-like
        Number of times each outcome occurred.
    base : float
        Logarithm base. Default is 2 (bits).

    Returns
    -------
    float
        Entropy of the empirical distribution implied by the counts.
    """
    counts = np.asarray(counts, dtype=np.float64)

    if np.any(counts < 0):
        raise ValueError("Counts must be non-negative.")

    # Convert counts to probabilities: p_i = count_i / total_count
    total = counts.sum()
    if total == 0:
        raise ValueError("Counts must sum to a positive value.")

    probabilities = counts / total
    return shannon_entropy(probabilities, base=base)


def max_entropy(num_outcomes: int, base: float = 2) -> float:
    """
    Return the maximum possible entropy for a uniform distribution
    over `num_outcomes` equally likely events.

    A fair die with 6 faces has max entropy = log₂(6) ≈ 2.58 bits.
    """
    if num_outcomes < 1:
        raise ValueError("Number of outcomes must be at least 1.")
    return float(np.log(num_outcomes) / np.log(base))


def describe_distribution(name: str, probabilities: np.ndarray) -> None:
    """Print a readable summary of a probability distribution and its entropy."""
    p = np.asarray(probabilities, dtype=np.float64)
    h = shannon_entropy(p)
    h_max = max_entropy(len(p))
    normalized = h / h_max if h_max > 0 else 0.0

    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")
    print(f"  Probabilities : {np.round(p, 4)}")
    print(f"  Entropy (bits)  : {h:.4f}")
    print(f"  Max possible    : {h_max:.4f} bits  (uniform over {len(p)} outcomes)")
    print(f"  Normalized      : {normalized:.1%} of maximum uncertainty")

    if normalized < 0.3:
        print("  → LOW entropy — outcomes are predictable / one dominates")
    elif normalized > 0.85:
        print("  → HIGH entropy — outcomes are nearly equally uncertain")
    else:
        print("  → MODERATE entropy — some structure, some uncertainty")


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE PROBABILITY DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────

def run_examples() -> None:
    """Demonstrate low vs high entropy with concrete probability distributions."""

    print("=" * 60)
    print("  EntropyShield-X — Shannon Entropy Calculator")
    print("=" * 60)
    print("\nEntropy measures average surprise (uncertainty) in a distribution.")
    print("Formula: H = - Σ p(x) · log₂(p(x))   [result in bits when base=2]")

    # ── LOW ENTROPY EXAMPLES ──────────────────────────────────────────────
    # When one outcome is almost certain, entropy is low.

    # Example 1: Very confident prediction (99% vs 1%)
    describe_distribution(
        "LOW ENTROPY — Confident binary classifier (spam filter)",
        [0.99, 0.01],
    )

    # Example 2: Skewed die — heavily favors one face
    describe_distribution(
        "LOW ENTROPY — Loaded die (one face dominates)",
        [0.70, 0.10, 0.08, 0.06, 0.04, 0.02],
    )

    # Example 3: Degenerate case — zero uncertainty (entropy = 0)
    describe_distribution(
        "LOW ENTROPY — Certain outcome (no surprise at all)",
        [1.0, 0.0, 0.0],
    )

    # ── HIGH ENTROPY EXAMPLES ─────────────────────────────────────────────
    # When outcomes are equally likely, entropy is maximized.

    # Example 4: Fair coin — maximum uncertainty for 2 outcomes
    describe_distribution(
        "HIGH ENTROPY — Fair coin flip (50/50)",
        [0.50, 0.50],
    )

    # Example 5: Fair six-sided die — uniform over 6 outcomes
    describe_distribution(
        "HIGH ENTROPY — Fair die (uniform over 6 faces)",
        [1 / 6] * 6,
    )

    # Example 6: Uniform over 8 outcomes (like random byte if all values equally likely)
    describe_distribution(
        "HIGH ENTROPY — Uniform byte distribution (encryption-like)",
        [1 / 8] * 8,
    )

    # ── AI / SECURITY SCENARIO ────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  AI SECURITY SCENARIO — Model confidence monitoring")
    print(f"{'─' * 60}")

    # Normal traffic: model confidently classifies as benign
    normal_logits_entropy = shannon_entropy([0.95, 0.03, 0.02])  # benign, suspicious, attack

    # Adversarial or out-of-distribution input: model becomes confused
    adversarial_entropy = shannon_entropy([0.40, 0.35, 0.25])

    print(f"  Normal request  → entropy = {normal_logits_entropy:.4f} bits  (confident)")
    print(f"  Adversarial/OOD → entropy = {adversarial_entropy:.4f} bits  (uncertain ⚠)")
    print("\n  Security insight: If entropy spikes above a threshold, flag for review.")

    # ── COUNT-BASED EXAMPLE ─────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  COUNT-BASED ENTROPY — Login IP diversity")
    print(f"{'─' * 60}")

    # Low entropy: same IP logs in repeatedly (normal user)
    normal_login_ips = np.array([98, 1, 1])  # counts per IP bucket
    # High entropy: logins spread across many IPs (possible credential stuffing)
    suspicious_login_ips = np.array([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

    h_normal = entropy_from_counts(normal_login_ips)
    h_suspicious = entropy_from_counts(suspicious_login_ips)

    print(f"  Normal user IPs     → entropy = {h_normal:.4f} bits")
    print(f"  Distributed attack  → entropy = {h_suspicious:.4f} bits")
    print("\n  Higher IP diversity entropy can indicate distributed brute-force attacks.")


if __name__ == "__main__":
    run_examples()
