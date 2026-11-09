"""Unit tests for k-NN and 5-fold out-of-fold label-auditing defenses in src/defense.py."""

import inspect
import pytest
import numpy as np
import pandas as pd

from src.data import load_spambase, split_data
from src.attacks import random_label_flip, targeted_spam_to_ham
from src.defense import (
    compute_knn_disagreement_scores,
    knn_label_audit,
    compute_oof_logreg_scores,
    oof_logreg_label_audit,
    evaluate_defense_auditing,
    compute_recovery_ratio,
)


@pytest.fixture(scope="module")
def spambase_split():
    """Load Spambase and return stratified 80/20 split for seed 0."""
    df = load_spambase()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


# ==============================================================================
# Week 2 Tests: k-NN Defense
# ==============================================================================

def test_knn_self_neighbor_exclusion(spambase_split):
    """Verify that no training example ever appears in its own k-NN neighborhood,
    even when duplicate feature vectors exist in the dataset."""
    X_train, _, y_train, _ = spambase_split
    y_poisoned, _ = targeted_spam_to_ham(y_train, poison_rate=0.10, random_state=0)

    scores, neighbor_indices, _ = compute_knn_disagreement_scores(X_train, y_poisoned, k=10)

    n_train = len(X_train)
    assert neighbor_indices.shape == (n_train, 10)
    assert scores.shape == (n_train,)

    for i in range(n_train):
        assert i not in neighbor_indices[i], f"Sample {i} was included in its own neighborhood!"
        assert len(np.unique(neighbor_indices[i])) == 10, f"Sample {i} does not have 10 unique neighbors!"


def test_knn_exact_score_calculation():
    """Verify exact k-NN disagreement score calculation on a controlled synthetic dataset."""
    # Construct 12 points in 2D:
    # First 11 points at (0, 0) + tiny offset: 10 labeled 0, 1 poisoned point (index 0) labeled 1
    # Plus 1 point far away
    rng = np.random.RandomState(42)
    X_cluster = rng.normal(loc=0.0, scale=0.001, size=(11, 2))
    X_far = np.array([[100.0, 100.0]])
    X_syn = np.vstack([X_cluster, X_far])

    # Point 0 has recorded label 1, while its 10 nearest other neighbors (indices 1..10) all have label 0
    # Point 1 has recorded label 0; among its 10 nearest neighbors (0, 2..10), 1 has label 1 and 9 have label 0 -> score = 0.1
    y_syn = np.array([1] + [0] * 10 + [0])

    scores, nbrs, _ = compute_knn_disagreement_scores(X_syn, y_syn, k=10)

    assert 0 not in nbrs[0]
    assert set(nbrs[0].tolist()) == set(range(1, 11))
    assert np.isclose(scores[0], 1.0), f"Expected score 1.0 for flipped point 0, got {scores[0]}"
    assert np.isclose(scores[1], 0.1), f"Expected score 0.1 for neighbor point 1, got {scores[1]}"

    audit_res = knn_label_audit(X_syn, y_syn, k=10, threshold=0.5)
    assert audit_res["flagged_mask"][0] is np.True_ or bool(audit_res["flagged_mask"][0]) is True
    assert int(audit_res["flagged_mask"].sum()) == 1
    assert len(audit_res["X_train_defended"]) == 11


def test_knn_reproducibility(spambase_split):
    """Verify that k-NN auditing produces bit-for-bit identical scores and masks across repeated runs."""
    X_train, _, y_train, _ = spambase_split
    y_poisoned, _ = targeted_spam_to_ham(y_train, poison_rate=0.10, random_state=0)

    res1 = knn_label_audit(X_train, y_poisoned, k=10, threshold=0.5)
    res2 = knn_label_audit(X_train, y_poisoned, k=10, threshold=0.5)

    assert np.array_equal(res1["suspicion_scores"], res2["suspicion_scores"])
    assert np.array_equal(res1["flagged_mask"], res2["flagged_mask"])
    assert np.array_equal(res1["neighbor_indices"], res2["neighbor_indices"])


def test_knn_training_only_scaling(spambase_split):
    """Verify that StandardScaler inside k-NN auditing is fit strictly on X_train."""
    X_train, X_test, y_train, _ = spambase_split
    y_poisoned, _ = random_label_flip(y_train, poison_rate=0.05, random_state=0)

    res = knn_label_audit(X_train, y_poisoned, k=10, threshold=0.5)
    scaler = res["scaler"]

    expected_mean = X_train.to_numpy().mean(axis=0)
    expected_var = X_train.to_numpy().var(axis=0)

    assert scaler.n_samples_seen_ == len(X_train)
    assert np.allclose(scaler.mean_, expected_mean, atol=1e-10)
    assert np.allclose(scaler.var_, expected_var, atol=1e-10)


def test_knn_test_set_and_clean_label_isolation(spambase_split):
    """Verify that k-NN auditing does not accept or mutate test data or clean labels."""
    X_train, X_test, y_train, y_test = spambase_split
    y_poisoned, flipped_mask = targeted_spam_to_ham(y_train, poison_rate=0.10, random_state=0)

    sig = inspect.signature(knn_label_audit)
    for forbidden_param in ["X_test", "y_test", "y_clean", "flipped_mask"]:
        assert forbidden_param not in sig.parameters

    x_test_hash_before = pd.util.hash_pandas_object(X_test).sum()
    y_test_hash_before = pd.util.hash_pandas_object(y_test).sum()
    y_train_hash_before = pd.util.hash_pandas_object(y_train).sum()
    y_poison_hash_before = pd.util.hash_pandas_object(y_poisoned).sum()

    res = knn_label_audit(X_train, y_poisoned, k=10, threshold=0.5)
    det_metrics = evaluate_defense_auditing(res["flagged_mask"], flipped_mask)

    assert pd.util.hash_pandas_object(X_test).sum() == x_test_hash_before
    assert pd.util.hash_pandas_object(y_test).sum() == y_test_hash_before
    assert pd.util.hash_pandas_object(y_train).sum() == y_train_hash_before
    assert pd.util.hash_pandas_object(y_poisoned).sum() == y_poison_hash_before
    assert 0.0 <= det_metrics["detection_precision"] <= 1.0
    assert 0.0 <= det_metrics["detection_recall"] <= 1.0


# ==============================================================================
# Week 3 Tests: 5-Fold Out-of-Fold Logistic-Regression Defense
# ==============================================================================

def test_oof_exactly_one_prediction_per_training_example(spambase_split):
    """Prove that 5-fold OOF auditing assigns each training example exactly one out-of-fold prediction."""
    X_train, _, y_train, _ = spambase_split
    y_poisoned, _ = targeted_spam_to_ham(y_train, poison_rate=0.10, random_state=0)

    scores, oof_probs, assign_counts = compute_oof_logreg_scores(
        X_train, y_poisoned, n_splits=5, random_state=0
    )

    n_train = len(X_train)
    assert assign_counts.shape == (n_train,)
    assert np.all(assign_counts == 1), "Every training example must receive exactly 1 OOF prediction!"
    assert not np.isnan(oof_probs).any(), "No OOF probability can be NaN!"
    assert np.allclose(oof_probs.sum(axis=1), 1.0, atol=1e-8), "OOF probabilities must sum to 1.0!"

    # Verify suspicion score formula = 1 - probability assigned to recorded label
    y_int = y_poisoned.to_numpy().astype(int)
    expected_scores = 1.0 - oof_probs[np.arange(n_train), y_int]
    assert np.allclose(scores, expected_scores, atol=1e-12)


def test_oof_reproducibility_and_test_set_isolation(spambase_split):
    """Prove that 5-fold OOF auditing is reproducible and completely isolated from X_test and y_test."""
    X_train, X_test, y_train, y_test = spambase_split
    y_poisoned, flipped_mask = random_label_flip(y_train, poison_rate=0.10, random_state=0)

    sig = inspect.signature(oof_logreg_label_audit)
    for forbidden_param in ["X_test", "y_test", "y_clean", "flipped_mask"]:
        assert forbidden_param not in sig.parameters

    x_test_hash_before = pd.util.hash_pandas_object(X_test).sum()
    y_test_hash_before = pd.util.hash_pandas_object(y_test).sum()

    res1 = oof_logreg_label_audit(X_train, y_poisoned, n_splits=5, threshold=0.8, random_state=0)
    res2 = oof_logreg_label_audit(X_train, y_poisoned, n_splits=5, threshold=0.8, random_state=0)
    res_diff_seed = oof_logreg_label_audit(X_train, y_poisoned, n_splits=5, threshold=0.8, random_state=1)

    assert np.allclose(res1["suspicion_scores"], res2["suspicion_scores"])
    assert np.array_equal(res1["flagged_mask"], res2["flagged_mask"])
    assert not np.allclose(res1["suspicion_scores"], res_diff_seed["suspicion_scores"])

    # Verify threshold >= 0.8 rule
    assert np.array_equal(res1["flagged_mask"], res1["suspicion_scores"] >= 0.8)

    # Verify test-set hashes untouched
    assert pd.util.hash_pandas_object(X_test).sum() == x_test_hash_before
    assert pd.util.hash_pandas_object(y_test).sum() == y_test_hash_before

    det = evaluate_defense_auditing(res1["flagged_mask"], flipped_mask)
    assert det["n_flagged"] == int(res1["flagged_mask"].sum())
    assert det["n_true_detected"] + det["n_clean_removed"] == det["n_flagged"]


def test_recovery_ratio_calculation_and_edge_cases():
    """Verify performance recovery ratio formula including negative and zero-denominator cases."""
    # Partial positive recovery: clean=0.90, poisoned=0.50, defended=0.80 -> (0.80-0.50)/(0.90-0.50) = 0.75
    assert np.isclose(compute_recovery_ratio(0.80, 0.50, 0.90), 0.75)

    # Negative recovery (when defense degrades metric below poisoned): defended=0.40 -> -0.25
    assert np.isclose(compute_recovery_ratio(0.40, 0.50, 0.90), -0.25)

    # Undefined recovery when clean == poisoned: returns NaN
    assert np.isnan(compute_recovery_ratio(0.85, 0.90, 0.90))
