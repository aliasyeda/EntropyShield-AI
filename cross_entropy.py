"""
EntropyShield-X — Cross Entropy Engine
======================================

WHAT IS CROSS-ENTROPY? (Intuitive explanation)
------------------------------------------------
Cross-entropy measures how well a *predicted* probability distribution
matches the *true* (ground-truth) distribution.

Think of it like a penalty score:
  - LOW cross-entropy  → prediction closely matches the truth  (good model)
  - HIGH cross-entropy → prediction is far from the truth      (bad model)

Formula (discrete case):

    H(P, Q) = - Σ  P(x) · log Q(x)

Where:
  - P(x) = true probability of outcome x  (what actually happened)
  - Q(x) = predicted probability of outcome x  (what the model guessed)
  - log  = logarithm (usually natural log in deep learning, log₂ in theory)

For classification with a single correct label (one-hot encoding),
only one term survives:

    Loss = -log( Q(correct_class) )

So if the model assigns 99% to the correct class → loss ≈ 0.01  (tiny)
If the model assigns  1% to the correct class → loss ≈ 4.61  (huge)

ENTROPY vs CROSS-ENTROPY
------------------------
  Entropy H(P)         = -Σ P(x) log P(x)   → uncertainty IN the true data itself
  Cross-entropy H(P,Q) = -Σ P(x) log Q(x)   → cost of using predictions Q for truth P

Relationship:
  Cross-entropy = Entropy + KL Divergence
  H(P, Q) = H(P) + D_KL(P || Q)

When Q = P (perfect prediction), cross-entropy equals entropy (minimum possible).

WHY LOGARITHMS ARE USED
-----------------------
1. HEAVY PENALTY FOR CONFIDENT WRONG ANSWERS
   log(0.99) ≈ -0.01   (small penalty — correct and confident)
   log(0.01) ≈ -4.61   (large penalty — wrong but confident)
   The log curve punishes low probabilities on the true class exponentially.

2. MATHEMATICAL CONVENIENCE
   Probabilities multiply; logs turn products into sums — easier gradients.

3. INFORMATION-THEORY FOUNDATION
   -log(p) is the "surprise" (self-information) of an event with probability p.
   Cross-entropy is the average surprise when reality follows P but you bet on Q.

WHY CROSS-ENTROPY IN NEURAL NETWORKS, TRANSFORMERS & LLMs
---------------------------------------------------------
Every classification head and language-model output layer produces a
probability distribution (via softmax). Training minimizes cross-entropy:

  - Image classifier: true label "cat" → maximize P(cat), minimize -log P(cat)
  - Transformer next-token prediction: true token "hello" → minimize
    -log P("hello" | context)
  - LLM pre-training: for each of billions of tokens, the model learns to
    assign high probability to the actual next word — that IS cross-entropy loss.

Why LLMs minimize cross-entropy during training:
  → It directly teaches the model to assign high probability to correct tokens.
  → Gradients flow cleanly through softmax + log (standard backprop).
  → It is equivalent to maximum-likelihood estimation — the statistically
    optimal way to fit a probabilistic model to data.

AI SECURITY CONNECTION
----------------------
Cross-entropy / loss patterns reveal security-relevant signals:

  - OVERCONFIDENT WRONG PREDICTIONS: A model that says 99% "benign" on malware
    has catastrophically high loss — and is dangerous if deployed without
    monitoring. Confidence ≠ correctness.

  - ADVERSARIAL ATTACKS: Crafted inputs push the model toward wrong but
    confident outputs. Monitoring cross-entropy on validation sets detects
    distribution shift and adversarial success.

  - ANOMALY DETECTION: Inputs with unusually high loss (OOD samples) often
    indicate attacks, poisoned data, or prompt-injection attempts the model
    never saw during training.
"""

import numpy as np


def cross_entropy(
    true_probs: np.ndarray,
    predicted_probs: np.ndarray,
    base: str = "natural",
) -> float:
    """
    Compute cross-entropy H(P, Q) = -Σ P(x) log Q(x).

    Parameters
    ----------
    true_probs : array-like
        Ground-truth probability distribution P. Must be non-negative.
    predicted_probs : array-like
        Model's predicted distribution Q. Must be non-negative.
    base : str
        "natural" (ln, used in PyTorch/TensorFlow) or "2" (bits).

    Returns
    -------
    float
        Cross-entropy value (lower is better).
    """
    # Convert both inputs to NumPy float arrays
    p = np.asarray(true_probs, dtype=np.float64)
    q = np.asarray(predicted_probs, dtype=np.float64)

    # Both distributions must have the same length
    if p.shape != q.shape:
        raise ValueError("true_probs and predicted_probs must have the same shape.")

    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Probabilities must be non-negative.")

    # Normalize P and Q so each sums to 1.0
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum == 0 or q_sum == 0:
        raise ValueError("Probability sums must be positive.")
    p = p / p_sum
    q = q / q_sum

    # Avoid log(0): clip predicted probs to a tiny floor value
    # (standard practice in ML frameworks for zero-probability predictions)
    q = np.clip(q, 1e-15, 1.0)

    # Pick log base: natural log for deep learning, log₂ for information theory
    if base == "2":
        log_q = np.log2(q)
    else:
        log_q = np.log(q)

    # H(P, Q) = - Σ P(x) log Q(x)
    # Only terms where P(x) > 0 contribute (0 · log q = 0)
    mask = p > 0
    ce = -np.sum(p[mask] * log_q[mask])

    return float(ce)


def categorical_cross_entropy(
    true_label: int,
    predicted_probs: np.ndarray,
    class_names: list[str] | None = None,
    base: str = "natural",
) -> float:
    """
    Cross-entropy for a single one-hot classification label.

    When the true label is class k, loss = -log Q(k).

    Parameters
    ----------
    true_label : int
        Index of the correct class (0-based).
    predicted_probs : array-like
        Model softmax output — one probability per class.
    class_names : list of str, optional
        Human-readable names for each class (for display only).
    base : str
        "natural" or "2".

    Returns
    -------
    float
        Cross-entropy loss for this single example.
    """
    q = np.asarray(predicted_probs, dtype=np.float64)

    if true_label < 0 or true_label >= len(q):
        raise ValueError(f"true_label {true_label} out of range [0, {len(q)}).")

    # Build one-hot true distribution: 1.0 at correct class, 0 elsewhere
    p = np.zeros_like(q)
    p[true_label] = 1.0

    return cross_entropy(p, q, base=base)


def binary_cross_entropy(
    true_label: int,
    predicted_prob_positive: float,
    base: str = "natural",
) -> float:
    """
    Cross-entropy for binary classification (label 0 or 1).

    Loss = -[ y·log(p) + (1-y)·log(1-p) ]
    """
    y = float(true_label)
    p = float(np.clip(predicted_prob_positive, 1e-15, 1 - 1e-15))

    if base == "2":
        log_fn = np.log2
    else:
        log_fn = np.log

    loss = -(y * log_fn(p) + (1 - y) * log_fn(1 - p))
    return float(loss)


def describe_prediction(
    scenario: str,
    true_label: int,
    predicted_probs: np.ndarray,
    class_names: list[str],
) -> dict:
    """
    Print a readable breakdown of one classification example and its loss.

    Returns a dict with loss values and metadata for programmatic use.
    """
    q = np.asarray(predicted_probs, dtype=np.float64)
    q = q / q.sum()  # ensure valid distribution

    loss_natural = categorical_cross_entropy(true_label, q, base="natural")
    loss_bits = categorical_cross_entropy(true_label, q, base="2")
    predicted_class = int(np.argmax(q))
    confidence = float(q[predicted_class])
    true_prob = float(q[true_label])

    correct = predicted_class == true_label

    print(f"\n{'-' * 62}")
    print(f"  {scenario}")
    print(f"{'-' * 62}")
    print(f"  True label      : {class_names[true_label]}  (index {true_label})")
    print(f"  Predicted label : {class_names[predicted_class]}  (index {predicted_class})")
    print(f"  Model confidence: {confidence:.1%} on predicted class")
    print(f"  P(true class)   : {true_prob:.4f}  <- this drives the loss")
    print(f"  Correct?        : {'YES [OK]' if correct else 'NO [X]'}")
    print(f"  Cross-entropy   : {loss_natural:.4f} nats  |  {loss_bits:.4f} bits")

    # Interpret loss magnitude for the learner
    if loss_natural < 0.1:
        verdict = "LOW LOSS - strong, correct prediction"
    elif loss_natural < 0.7:
        verdict = "MODERATE LOSS - uncertain or partially correct"
    else:
        verdict = "HIGH LOSS - wrong and/or overconfident on wrong class"

    print(f"  Verdict         : {verdict}")

    # Per-class probability table
    print(f"\n  {'Class':<14} {'Probability':>12} {'Bar'}")
    for i, name in enumerate(class_names):
        bar_len = int(q[i] * 30)
        marker = " <- true" if i == true_label else ""
        print(f"  {name:<14} {q[i]:>11.2%}  {'#' * bar_len}{marker}")

    return {
        "loss_natural": loss_natural,
        "loss_bits": loss_bits,
        "correct": correct,
        "confidence": confidence,
        "true_prob": true_prob,
    }


def run_examples() -> None:
    """Run low-loss, high-loss, and security-focused cross-entropy demos."""

    print("=" * 62)
    print("  EntropyShield-X - Cross Entropy Engine")
    print("=" * 62)
    print("\nCross-entropy = - sum( P(true) * log Q(predicted) )")
    print("For one-hot labels:  Loss = -log( P assigned to correct class )")
    print("Lower loss = better match between prediction and truth.\n")

    # Shared label set for a network-intrusion classifier
    classes = ["benign", "suspicious", "attack"]

    # ── SECTION 1: LOW LOSS — Correct confident prediction ──────────────
    print("\n" + "=" * 62)
    print("  SECTION 1 - LOW LOSS: Correct & Confident Prediction")
    print("=" * 62)
    print("  The model correctly identifies benign traffic with 97% confidence.")
    print("  Loss = -log(0.97) ~= 0.03 - tiny penalty, training signal says 'keep doing this'.")

    describe_prediction(
        scenario="[OK] Correct confident prediction (spam filter: NOT spam)",
        true_label=0,  # benign
        predicted_probs=[0.97, 0.02, 0.01],
        class_names=classes,
    )

    # ── SECTION 2: HIGH LOSS — Wrong confident prediction ─────────────────
    print("\n" + "=" * 62)
    print("  SECTION 2 - HIGH LOSS: Wrong & Overconfident Prediction")
    print("=" * 62)
    print("  The model says 96% benign, but the true label is ATTACK.")
    print("  Loss = -log(0.02) ~= 3.91 - massive penalty from the logarithm.")
    print("  Why log? Because log(0.02) is very negative -> -log is very POSITIVE.")

    describe_prediction(
        scenario="[X] Wrong confident prediction (missed malware)",
        true_label=2,  # attack
        predicted_probs=[0.96, 0.03, 0.01],
        class_names=classes,
    )

    # ── SECTION 3: MODERATE LOSS — Uncertain prediction ─────────────────
    print("\n" + "=" * 62)
    print("  SECTION 3 - MODERATE LOSS: Uncertain Prediction")
    print("=" * 62)
    print("  The model is unsure - probabilities are spread across classes.")
    print("  Loss is moderate: the model didn't confidently choose wrong,")
    print("  but also didn't confidently choose right.")

    describe_prediction(
        scenario="~ Uncertain but correct prediction (needs human review)",
        true_label=1,  # suspicious
        predicted_probs=[0.25, 0.40, 0.35],  # right class, low confidence
        class_names=classes,
    )

    # ── SECTION 4: LLM next-token example ───────────────────────────────
    print("\n" + "=" * 62)
    print("  SECTION 4 - LLM Training: Next-Token Cross-Entropy")
    print("=" * 62)

    vocab = ["the", "attack", "hello", "firewall"]
    # Context: "... detected an" → true next token is "attack"
    llm_probs_good = [0.05, 0.85, 0.05, 0.05]   # model predicts "attack" well
    llm_probs_bad = [0.60, 0.15, 0.15, 0.10]    # model misses "attack"

    good_loss = categorical_cross_entropy(1, llm_probs_good)
    bad_loss = categorical_cross_entropy(1, llm_probs_bad)

    print("  LLMs predict the next token from a vocabulary of thousands/millions.")
    print("  Training minimizes cross-entropy over every token in the corpus.\n")
    print(f"  Context ends with: '...detected an ___'")
    print(f"  True next token   : '{vocab[1]}'")
    print(f"  Good model P(token): {llm_probs_good[1]:.0%}  -> loss = {good_loss:.4f}")
    print(f"  Bad  model P(token): {llm_probs_bad[1]:.0%}  -> loss = {bad_loss:.4f}")
    print("\n  Billions of these tiny loss computations = how GPT/LLaMA learn language.")

    # ── SECTION 5: Binary example (phishing detector) ───────────────────
    print("\n" + "=" * 62)
    print("  SECTION 5 - Binary Cross-Entropy (Phishing Detector)")
    print("=" * 62)

    # True label 1 = phishing; model outputs P(phishing)
    bce_correct = binary_cross_entropy(true_label=1, predicted_prob_positive=0.92)
    bce_wrong = binary_cross_entropy(true_label=1, predicted_prob_positive=0.08)

    print("  Binary CE: -[ y*log(p) + (1-y)*log(1-p) ]")
    print(f"  Phishing email, model says 92% phishing -> loss = {bce_correct:.4f}  (low)")
    print(f"  Phishing email, model says  8% phishing -> loss = {bce_wrong:.4f}  (high)")

    # ── SECTION 6: Full distribution cross-entropy ────────────────────────
    print("\n" + "=" * 62)
    print("  SECTION 6 - Full Distribution Cross-Entropy (P vs Q)")
    print("=" * 62)

    # Soft labels: 60% attack, 40% suspicious (expert-annotated ambiguity)
    true_soft = np.array([0.0, 0.40, 0.60])
    pred_sharp = np.array([0.05, 0.10, 0.85])   # model overconfident on attack
    pred_calibrated = np.array([0.05, 0.38, 0.57])  # model closer to truth

    ce_sharp = cross_entropy(true_soft, pred_sharp)
    ce_calibrated = cross_entropy(true_soft, pred_calibrated)

    print("  When labels are soft (not one-hot), all classes contribute to loss.")
    print(f"  True distribution     : {true_soft}")
    print(f"  Overconfident model   : {pred_sharp}  -> CE = {ce_sharp:.4f}")
    print(f"  Calibrated model      : {pred_calibrated}  -> CE = {ce_calibrated:.4f}")

    # ── AI SECURITY INSIGHT ───────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  AI SECURITY INSIGHT")
    print("=" * 62)
    print("""
  1. OVERCONFIDENT WRONG PREDICTIONS ARE DANGEROUS
     A firewall ML model that labels malware as 'benign' with 96% confidence
     will NOT trigger human review - the high confidence silences alarms.
     Cross-entropy loss on audit logs reveals these silent failures.

  2. ADVERSARIAL ATTACKS EXPLOIT THE LOSS SURFACE
     Attackers craft inputs (adversarial examples, prompt injections) that
     flip predictions while staying near normal data. Monitoring per-sample
     cross-entropy on production traffic detects inputs the model struggles with.

  3. ANOMALY DETECTION VIA LOSS THRESHOLDS
     Out-of-distribution inputs (novel attacks, zero-day patterns) typically
     produce HIGH cross-entropy because the model assigns low probability to
     all known classes. Flag samples where loss > threshold for investigation.

  4. WHY LLMs MINIMIZE CROSS-ENTROPY
     Every training step adjusts billions of weights to reduce -log P(correct token).
     Lower cross-entropy = model assigns more probability mass to what humans
     actually wrote - which is exactly 'learning to predict language'.

  Key takeaway: Cross-entropy connects mathematical training (gradients),
  information theory (surprise), and security monitoring (loss spikes).
""")

    # Summary comparison table
    print("-" * 62)
    print("  QUICK COMPARISON - Same true label (attack), different predictions")
    print("-" * 62)
    scenarios = [
        ("Correct + confident", [0.01, 0.02, 0.97]),
        ("Wrong + confident",   [0.97, 0.02, 0.01]),
        ("Uncertain",           [0.33, 0.34, 0.33]),
    ]
    print(f"  {'Scenario':<24} {'P(attack)':>10} {'Loss (nats)':>12}")
    for name, probs in scenarios:
        loss = categorical_cross_entropy(2, probs)
        print(f"  {name:<24} {probs[2]:>9.0%} {loss:>12.4f}")
    print("-" * 62)


if __name__ == "__main__":
    run_examples()
