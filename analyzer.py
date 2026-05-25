#!/usr/bin/env python3
"""
EntropyShield-X - KL Divergence Analyzer
========================================

Educational research-style tool for learning KL divergence and its role in
AI, transformers, cybersecurity, and machine learning.

Run:
    python analyzer.py

Dependencies: Python 3.10+, NumPy only.
"""

from __future__ import annotations

import numpy as np

from cross_entropy import cross_entropy
from display import banner, bar, print_distribution_table, print_formula_block, subheader, verdict_kl
from entropy import shannon_entropy
from kl_divergence import kl_divergence, kl_symmetry_gap, verify_cross_entropy_identity


# ─────────────────────────────────────────────────────────────────────────────
# 1. COMPARE TWO DISTRIBUTIONS (reusable demo helper)
# ─────────────────────────────────────────────────────────────────────────────

def compare_distributions(
    scenario: str,
    true_probs: np.ndarray,
    predicted_probs: np.ndarray,
    labels: list[str],
    *,
    show_identity: bool = True,
) -> dict:
    """
    Compare ground-truth P with prediction Q: print tables and return metrics.

    This is the main teaching function — use it for every scenario block.
    """
    p = np.asarray(true_probs, dtype=np.float64)
    q = np.asarray(predicted_probs, dtype=np.float64)

    kl_nat = kl_divergence(p, q, base="natural")
    kl_bits = kl_divergence(p, q, base="2")
    ce = cross_entropy(p, q, base="natural")
    h_p = shannon_entropy(p, base=np.e)

    subheader(scenario)
    print_formula_block([
        "D_KL(P || Q) = sum P(x) * log( P(x) / Q(x) )",
        "P = true (reality)     Q = predicted (model / observed)",
    ])

    print_distribution_table(labels, p, q, kl_nat, unit="nats")

    print(f"\n  Entropy H(P)           : {h_p:.4f} nats")
    print(f"  Cross-entropy H(P,Q)   : {ce:.4f} nats")
    print(f"  KL divergence          : {kl_nat:.4f} nats  |  {kl_bits:.4f} bits")
    print(f"  Interpretation         : {verdict_kl(kl_nat)}")

    if show_identity:
        ident = verify_cross_entropy_identity(p, q)
        print(f"\n  Identity check:  H(P,Q) = H(P) + KL")
        print(f"    H(P,Q) = {ident['cross_entropy_H_PQ']:.6f}")
        print(f"    H(P)+KL = {ident['sum_H_plus_KL']:.6f}  (residual {ident['residual']:.2e})")

    # Visual bars
    print(f"\n  {'Label':<14} {'P (true)':<28} {'Q (pred)':<28}")
    for i, name in enumerate(labels):
        print(f"  {name:<14} {bar(p[i])}  {bar(q[i])}")

    return {"kl_nats": kl_nat, "kl_bits": kl_bits, "cross_entropy": ce, "entropy": h_p}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EDUCATIONAL PROGRAM
# ─────────────────────────────────────────────────────────────────────────────

def print_intro() -> None:
    banner("EntropyShield-X - KL Divergence Analyzer", 72)
    print("""
  A beginner-friendly research console for understanding how probability
  distributions differ - and why that matters for AI, LLMs, and security.

  +----------------------------------------------------------------------+
  |  CORE FORMULA                                                        |
  |                                                                      |
  |    D_KL(P || Q)  =  sum  P(x) * log( P(x) / Q(x) )                  |
  |                                                                      |
  |    P = TRUE distribution (reality, ground truth)                     |
  |    Q = PREDICTED distribution (model, sensor, attacker profile)      |
  +----------------------------------------------------------------------+

  What KL tells you:
    * KL = 0     when P and Q are identical (perfect match)
    * KL > 0     always - never negative
    * Larger KL  -> Q is a worse substitute for P
    * NOT symmetric: D_KL(P||Q) != D_KL(Q||P)

  The bridge to machine learning:
    Cross-Entropy  =  Entropy  +  KL Divergence
    H(P, Q)        =  H(P)     +  D_KL(P || Q)
""")


def section_entropy_cross_entropy_kl() -> None:
    """Explain the three quantities and verify the identity numerically."""
    banner("RELATIONSHIP: Entropy | Cross-Entropy | KL Divergence", 72)

    print_formula_block([
        "Entropy H(P)         = -sum P(x) log P(x)",
        "  -> Average surprise IN the true data itself (uncertainty of reality).",
        "",
        "Cross-entropy H(P,Q) = -sum P(x) log Q(x)",
        "  -> Average surprise when truth is P but you encode/bet with Q.",
        "",
        "KL divergence        =  sum P(x) log( P(x)/Q(x) )",
        "  -> EXTRA surprise from using Q instead of the ideal P.",
        "",
        "Identity:  H(P,Q) = H(P) + D_KL(P||Q)",
        "  -> Cross-entropy is entropy plus the 'penalty' for wrong predictions.",
    ])

    # Example where Q is slightly off
    p = np.array([0.50, 0.30, 0.20])
    q = np.array([0.45, 0.35, 0.20])

    ident = verify_cross_entropy_identity(p, q)
    subheader("Numeric verification (3-class example)")
    print(f"  P (true)      : {p}")
    print(f"  Q (predicted) : {q}")
    print(f"  H(P)          : {ident['entropy_H_P']:.6f} nats")
    print(f"  D_KL(P||Q)    : {ident['kl_D_KL']:.6f} nats")
    print(f"  H(P) + KL     : {ident['sum_H_plus_KL']:.6f} nats")
    print(f"  H(P,Q)        : {ident['cross_entropy_H_PQ']:.6f} nats")
    print(f"  Residual      : {ident['residual']:.2e}  (should be ~0)")


def section_low_vs_high_divergence() -> None:
    """Side-by-side LOW vs HIGH KL examples."""
    banner("LOW vs HIGH KL DIVERGENCE", 72)

    labels = ["class_A", "class_B", "class_C"]
    p = np.array([0.70, 0.20, 0.10])

    # LOW: Q almost matches P
    q_good = np.array([0.68, 0.22, 0.10])
    # HIGH: Q is far from P
    q_bad = np.array([0.10, 0.10, 0.80])

    r_good = compare_distributions(
        "LOW DIVERGENCE - Good prediction (Q ~ P)",
        p, q_good, labels,
    )
    r_bad = compare_distributions(
        "HIGH DIVERGENCE - Bad prediction (Q far from P)",
        p, q_bad, labels,
    )

    subheader("Quick comparison")
    print(f"  Good model KL : {r_good['kl_nats']:.4f} nats")
    print(f"  Bad  model KL : {r_bad['kl_nats']:.4f} nats")
    print(f"  Ratio (bad/good): {r_bad['kl_nats'] / max(r_good['kl_nats'], 1e-9):.1f}x more divergence")


def section_kl_not_symmetric() -> None:
    """Demonstrate D_KL(P||Q) ≠ D_KL(Q||P)."""
    banner("KL IS NOT SYMMETRIC", 72)

    print_formula_block([
        "D_KL(P||Q) answers: 'How much extra cost if truth is P but I use Q?'",
        "D_KL(Q||P) answers: the reverse question - different answer!",
        "",
        "This is why KL is a 'divergence', not a distance metric.",
    ])

    p = np.array([0.90, 0.10])
    q = np.array([0.50, 0.50])

    kl_pq, kl_qp, gap = kl_symmetry_gap(p, q)
    print(f"\n  P (true)     : {p}  - reality is heavily skewed")
    print(f"  Q (predicted): {q}  - model acts like a fair coin")
    print(f"\n  D_KL(P || Q) = {kl_pq:.4f} nats  (truth skewed, model uniform)")
    print(f"  D_KL(Q || P) = {kl_qp:.4f} nats  (swap roles)")
    print(f"  |difference| = {gap:.4f} nats")


def section_ai_examples() -> None:
    """ML classifier: good vs bad predictions."""
    banner("AI / MACHINE LEARNING - Classifier Calibration", 72)

    classes = ["benign", "suspicious", "malware"]

    # True traffic mix in validation set
    p_true = np.array([0.80, 0.15, 0.05])

    compare_distributions(
        "Good AI - Calibrated security classifier",
        p_true,
        np.array([0.78, 0.17, 0.05]),
        classes,
    )

    compare_distributions(
        "Bad AI - Overconfident on wrong class (silent failure risk)",
        p_true,
        np.array([0.95, 0.03, 0.02]),  # says mostly benign; reality has more malware
        classes,
    )


def section_llm_examples() -> None:
    """Transformer next-token distributions."""
    banner("LLM / TRANSFORMER - Next-Token Distributions", 72)

    print_formula_block([
        "LLMs output Q = P(token | context) over the vocabulary.",
        "Training pushes Q toward the true next token (one-hot P).",
        "Per-token cross-entropy = -log Q(correct); KL to one-hot is the training signal.",
    ])

    vocab = ["the", "firewall", "detected", "breach"]

    # One-hot true token: "breach" (index 3)
    p_onehot = np.array([0.0, 0.0, 0.0, 1.0])

    compare_distributions(
        "Confident AI - High P(correct token), low KL to truth",
        p_onehot,
        np.array([0.02, 0.03, 0.05, 0.90]),
        vocab,
    )

    compare_distributions(
        "Confused AI - Flat Q, high KL (model uncertain / OOD input)",
        p_onehot,
        np.array([0.25, 0.25, 0.25, 0.25]),
        vocab,
    )

    subheader("Training intuition")
    print("  Billions of tokens x minimize cross-entropy ~ minimize KL to one-hot P.")
    print("  Lower KL per token -> model language matches human-written text.")


def section_security_examples() -> None:
    """Normal vs attack traffic profiles."""
    banner("CYBERSECURITY - Normal vs Attack Traffic Profiles", 72)

    print_formula_block([
        "P = baseline profile learned from normal network traffic.",
        "Q = live window of traffic (what we observe right now).",
        "KL(P||Q) spike -> distribution shift -> investigate for intrusion.",
    ])

    protocols = ["HTTPS", "DNS", "SSH", "other"]

    p_normal = np.array([0.55, 0.30, 0.10, 0.05])

    compare_distributions(
        "Normal traffic - Q matches baseline P (LOW KL)",
        p_normal,
        np.array([0.53, 0.31, 0.11, 0.05]),
        protocols,
    )

    compare_distributions(
        "Attack traffic - Exfiltration via DNS tunneling (HIGH KL)",
        p_normal,
        np.array([0.20, 0.55, 0.05, 0.20]),
        protocols,
    )

    subheader("Login behavior - IP source diversity")
    p_user = np.array([0.95, 0.03, 0.02])  # same user, same IP bucket
    q_attack = np.array([0.15, 0.15, 0.70])  # distributed brute force

    compare_distributions(
        "Credential stuffing - IP distribution Q vs normal user P",
        p_user,
        q_attack,
        ["home_IP", "corp_VPN", "unknown_IP"],
    )


def section_adversarial() -> None:
    """Adversarial / OOD inputs and entropy + KL together."""
    banner("ADVERSARIAL & OUT-OF-DISTRIBUTION (OOD) BEHAVIOR", 72)

    print_formula_block([
        "Adversarial example: input crafted so model Q looks confident but is WRONG.",
        "OOD sample: input unlike training data -> Q becomes flat (high entropy).",
        "",
        "Monitor BOTH:",
        "  * KL(P||Q) or cross-entropy vs known truth (evaluation / labeled audit)",
        "  * Entropy of Q on unlabeled production traffic (confidence collapse)",
    ])

    classes = ["benign", "attack"]
    p_truth = np.array([0.0, 1.0])  # true label: attack

    # Adversarial: wrong but confident benign
    q_adv = np.array([0.98, 0.02])
    r = compare_distributions(
        "Adversarial - Model confident WRONG (dangerous in production)",
        p_truth,
        q_adv,
        classes,
    )

    # OOD: confused uniform-ish
    q_ood = np.array([0.48, 0.52])
    compare_distributions(
        "OOD / confused model - Uncertain Q (needs human review)",
        p_truth,
        q_ood,
        classes,
    )

    h_confident = shannon_entropy(q_adv, base=np.e)
    h_confused = shannon_entropy(q_ood, base=np.e)

    subheader("Entropy of model output Q (confidence monitor)")
    print(f"  Adversarial Q entropy : {h_confident:.4f} nats  (LOW - looks confident)")
    print(f"  OOD Q entropy         : {h_confused:.4f} nats  (HIGH - model unsure)")
    print(f"  Cross-entropy (adv.)  : {r['cross_entropy']:.4f} nats  (HIGH loss despite confidence)")
    print("\n  Security lesson: confidence alone does not prove correctness.")
    print("  Combine KL/CE loss spikes with entropy and input metadata.")


def section_interpretation() -> None:
    """How to read KL values in practice."""
    banner("OUTPUT INTERPRETATION GUIDE", 72)

    print("""
  +------------------+--------------------------------------------------------+
  | KL (nats)        | Meaning                                                |
  +------------------+--------------------------------------------------------+
  | ~ 0              | P ~ Q - distributions match (ideal / normal)         |
  | 0.01 - 0.10      | Small drift - acceptable calibration noise             |
  | 0.10 - 0.50      | Noticeable mismatch - tune model or investigate        |
  | > 0.50           | Strong mismatch - bad predictions or possible anomaly  |
  | -> inf (theory)  | Q(x)=0 where P(x)>0 - zero prob on a real event        |
  +------------------+--------------------------------------------------------+

  How to use this tool:
    1. Define P = what you believe is true (data, baseline, one-hot label).
    2. Define Q = what the system predicts or what you observe live.
    3. Run compare_distributions() or kl_divergence(P, Q).
    4. Check Cross-Entropy = Entropy + KL for consistency.
    5. In security, set alerts when KL exceeds a baseline learned from P.

  Remember:
    * KL(P||Q) uses P as truth - order matters.
    * For classifiers, P is often one-hot (single correct class).
    * For traffic analytics, P and Q are both histograms over features.
    * Lower cross-entropy <=> lower KL (when H(P) is fixed for the same P).
""")


def section_summary_table() -> None:
    """Final comparison table across all demo scenarios."""
    banner("SUMMARY - All Demo Scenarios", 72)

    rows = [
        ("Good classifier", [0.8, 0.15, 0.05], [0.78, 0.17, 0.05]),
        ("Bad classifier", [0.8, 0.15, 0.05], [0.95, 0.03, 0.02]),
        ("Normal traffic", [0.55, 0.30, 0.10, 0.05], [0.53, 0.31, 0.11, 0.05]),
        ("Attack traffic", [0.55, 0.30, 0.10, 0.05], [0.20, 0.55, 0.05, 0.20]),
        ("LLM confident", [0, 0, 0, 1], [0.02, 0.03, 0.05, 0.90]),
        ("LLM confused", [0, 0, 0, 1], [0.25, 0.25, 0.25, 0.25]),
    ]

    print(f"\n  {'Scenario':<22} {'KL (nats)':>12} {'CE':>12} {'Verdict'}")
    print(f"  {'-' * 68}")
    for name, p, q in rows:
        p_arr, q_arr = np.array(p, dtype=np.float64), np.array(q, dtype=np.float64)
        kl = kl_divergence(p_arr, q_arr)
        ce = cross_entropy(p_arr, q_arr)
        print(f"  {name:<22} {kl:>12.4f} {ce:>12.4f}  {verdict_kl(kl)[:40]}...")
    print(f"  {'-' * 68}")
    print("\n  EntropyShield-X session complete. Re-run: python analyzer.py\n")


def _configure_stdout() -> None:
    """Use UTF-8 on stdout when the terminal supports it (avoids Windows cp1252 errors)."""
    try:
        import sys
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main() -> None:
    """Run the full educational KL divergence analysis program."""
    _configure_stdout()
    print_intro()
    section_entropy_cross_entropy_kl()
    section_low_vs_high_divergence()
    section_kl_not_symmetric()
    section_ai_examples()
    section_llm_examples()
    section_security_examples()
    section_adversarial()
    section_interpretation()
    section_summary_table()


if __name__ == "__main__":
    main()
