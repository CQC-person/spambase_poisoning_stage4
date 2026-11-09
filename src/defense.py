"""Manual label-auditing defense implementations for Stage 4:
1. Primary defense: k-NN neighborhood-disagreement auditing (k=10, threshold=0.5)
2. Extended defense: 5-fold out-of-fold (OOF) logistic-regression auditing (threshold=0.8)

All functions use only NumPy, Pandas, and scikit-learn primitives and enforce strict
training-only scaling, self-neighbor exclusion, and test-set / clean-label isolation.
"""

import time
from typing import Any, Dict, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression


ArrayLikeX = Union[pd.DataFrame, np.ndarray]
ArrayLikeY = Union[pd.Series, np.ndarray]


def compute_knn_disagreement_scores(
    X_train: ArrayLikeX,
    y_train_recorded: ArrayLikeY,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Compute self-excluded k-NN label disagreement scores on poisoned training data.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training feature matrix of shape (N_train, d).
    y_train_recorded : pd.Series or np.ndarray
        Recorded (potentially poisoned) binary training labels of shape (N_train,).
    k : int, default=10
        Number of nearest other training examples to query.

    Returns
    -------
    suspicion_scores : np.ndarray of shape (N_train,)
        Fraction of the k nearest other training examples whose recorded labels
        disagree with each example's recorded label.
    neighbor_indices : np.ndarray of shape (N_train, k)
        Indices of the k nearest other training examples for each training instance
        (guaranteed to exclude index i itself).
    scaler : StandardScaler
        StandardScaler fit strictly on X_train.
    """
    X_arr = X_train.to_numpy( copy=True ) if isinstance(X_train, pd.DataFrame) else np.array(X_train, copy=True)
    y_arr = y_train_recorded.to_numpy( copy=True ) if isinstance(y_train_recorded, pd.Series) else np.array(y_train_recorded, copy=True)

    n_samples = len(X_arr)
    if k < 1 or k >= n_samples:
        raise ValueError(f"k must be between 1 and n_samples - 1 ({n_samples - 1}), got {k}.")

    # 1. Fit StandardScaler strictly on poisoned training features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    # 2. Query k + 1 neighbors and explicitly exclude each sample itself
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="auto")
    nn.fit(X_scaled)
    _, raw_indices = nn.kneighbors(X_scaled)

    neighbor_indices = np.empty((n_samples, k), dtype=int)
    for i in range(n_samples):
        row_nbrs = raw_indices[i]
        # Explicitly filter out self index i even if duplicate feature rows exist
        other_nbrs = row_nbrs[row_nbrs != i]
        neighbor_indices[i] = other_nbrs[:k]

    # 3. Compute fraction of k neighbors whose recorded labels disagree with y_arr[i]
    neighbor_labels = y_arr[neighbor_indices]
    suspicion_scores = np.mean(neighbor_labels != y_arr[:, None], axis=1)

    return suspicion_scores, neighbor_indices, scaler


def knn_label_audit(
    X_train: ArrayLikeX,
    y_train_recorded: ArrayLikeY,
    k: int = 10,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Run primary k-NN neighborhood-disagreement label auditing and filter flagged samples.

    Clean labels and poisoning masks are never passed to or used by this function.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training feature matrix.
    y_train_recorded : pd.Series or np.ndarray
        Recorded (poisoned) training label vector.
    k : int, default=10
        Number of nearest other training neighbors.
    threshold : float, default=0.5
        Disagreement threshold at or above which samples are flagged and removed.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing suspicion_scores, flagged_mask, keep_mask,
        neighbor_indices, scaler, X_train_defended, y_train_defended, and audit_runtime_sec.
    """
    t0 = time.perf_counter()
    suspicion_scores, neighbor_indices, scaler = compute_knn_disagreement_scores(
        X_train, y_train_recorded, k=k
    )
    flagged_mask = suspicion_scores >= threshold
    keep_mask = ~flagged_mask

    if isinstance(X_train, pd.DataFrame):
        X_defended = X_train.iloc[keep_mask].copy()
    else:
        X_defended = X_train[keep_mask].copy()

    if isinstance(y_train_recorded, pd.Series):
        y_defended = y_train_recorded.iloc[keep_mask].copy()
    else:
        y_defended = y_train_recorded[keep_mask].copy()

    audit_runtime_sec = time.perf_counter() - t0

    return {
        "defense": "knn",
        "k": k,
        "threshold": threshold,
        "suspicion_scores": suspicion_scores,
        "flagged_mask": flagged_mask,
        "keep_mask": keep_mask,
        "neighbor_indices": neighbor_indices,
        "scaler": scaler,
        "X_train_defended": X_defended,
        "y_train_defended": y_defended,
        "audit_runtime_sec": audit_runtime_sec,
    }


def compute_oof_logreg_scores(
    X_train: ArrayLikeX,
    y_train_recorded: ArrayLikeY,
    n_splits: int = 5,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 5-fold out-of-fold standardized logistic-regression suspicion scores.

    Within each fold, StandardScaler and LogisticRegression are fit strictly on the
    other four folds and predict probabilities on the held-out fold, ensuring every
    training example receives exactly one out-of-fold probability estimate.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training feature matrix.
    y_train_recorded : pd.Series or np.ndarray
        Recorded (poisoned) binary training labels.
    n_splits : int, default=5
        Number of stratified cross-validation folds.
    random_state : int, default=0
        Experiment seed for StratifiedKFold shuffling and LogisticRegression.

    Returns
    -------
    suspicion_scores : np.ndarray of shape (N_train,)
        1 - out-of-fold probability assigned to each sample's recorded label.
    oof_probs : np.ndarray of shape (N_train, 2)
        Out-of-fold predicted probabilities [P(y=0|x), P(y=1|x)] for each training sample.
    oof_assign_counts : np.ndarray of shape (N_train,)
        Count of out-of-fold predictions received by each sample (all exactly 1).
    """
    X_arr = X_train.to_numpy(copy=True) if isinstance(X_train, pd.DataFrame) else np.array(X_train, copy=True)
    y_arr = y_train_recorded.to_numpy(copy=True) if isinstance(y_train_recorded, pd.Series) else np.array(y_train_recorded, copy=True)
    y_int = y_arr.astype(int)

    n_samples = len(X_arr)
    oof_probs = np.full((n_samples, 2), np.nan, dtype=float)
    oof_assign_counts = np.zeros(n_samples, dtype=int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_train_idx, fold_val_idx in skf.split(X_arr, y_int):
        scaler = StandardScaler()
        X_fold_train = scaler.fit_transform(X_arr[fold_train_idx])
        X_fold_val = scaler.transform(X_arr[fold_val_idx])

        clf = LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            random_state=random_state,
        )
        clf.fit(X_fold_train, y_int[fold_train_idx])
        fold_probs = clf.predict_proba(X_fold_val)

        oof_probs[fold_val_idx] = fold_probs
        oof_assign_counts[fold_val_idx] += 1

    assert np.all(oof_assign_counts == 1), "Every training example must receive exactly one OOF prediction."
    assert not np.isnan(oof_probs).any(), "OOF probability matrix contains NaNs."

    recorded_label_probs = oof_probs[np.arange(n_samples), y_int]
    suspicion_scores = 1.0 - recorded_label_probs

    return suspicion_scores, oof_probs, oof_assign_counts


def oof_logreg_label_audit(
    X_train: ArrayLikeX,
    y_train_recorded: ArrayLikeY,
    n_splits: int = 5,
    threshold: float = 0.8,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Run extended 5-fold out-of-fold logistic-regression label auditing and filter flagged samples.

    Clean labels and poisoning masks are never passed to or used by this function.
    """
    t0 = time.perf_counter()
    suspicion_scores, oof_probs, oof_assign_counts = compute_oof_logreg_scores(
        X_train, y_train_recorded, n_splits=n_splits, random_state=random_state
    )
    flagged_mask = suspicion_scores >= threshold
    keep_mask = ~flagged_mask

    if isinstance(X_train, pd.DataFrame):
        X_defended = X_train.iloc[keep_mask].copy()
    else:
        X_defended = X_train[keep_mask].copy()

    if isinstance(y_train_recorded, pd.Series):
        y_defended = y_train_recorded.iloc[keep_mask].copy()
    else:
        y_defended = y_train_recorded[keep_mask].copy()

    audit_runtime_sec = time.perf_counter() - t0

    return {
        "defense": "oof_logreg",
        "n_splits": n_splits,
        "threshold": threshold,
        "suspicion_scores": suspicion_scores,
        "oof_probs": oof_probs,
        "oof_assign_counts": oof_assign_counts,
        "flagged_mask": flagged_mask,
        "keep_mask": keep_mask,
        "X_train_defended": X_defended,
        "y_train_defended": y_defended,
        "audit_runtime_sec": audit_runtime_sec,
    }


def evaluate_defense_auditing(
    flagged_mask: np.ndarray,
    true_flipped_mask: np.ndarray,
) -> Dict[str, Union[int, float]]:
    """Evaluate label-auditing detection performance strictly post-hoc against ground-truth flipped_mask.

    Parameters
    ----------
    flagged_mask : np.ndarray of bool
        Boolean array indicating which training examples were flagged by the defense.
    true_flipped_mask : np.ndarray of bool
        Ground-truth boolean array indicating which training examples were actually poisoned.

    Returns
    -------
    Dict[str, Union[int, float]]
        Detection precision, recall, F1, number flagged, number/fraction of clean
        samples incorrectly removed, and number of true poisoned samples detected.
    """
    flagged = np.asarray(flagged_mask, dtype=bool)
    poisoned = np.asarray(true_flipped_mask, dtype=bool)
    clean = ~poisoned

    n_train = int(len(flagged))
    n_poisoned = int(poisoned.sum())
    n_clean_train = int(clean.sum())
    n_flagged = int(flagged.sum())

    tp_detect = int((flagged & poisoned).sum())
    fp_clean_removed = int((flagged & clean).sum())
    fn_missed_poison = int(((~flagged) & poisoned).sum())
    tn_clean_kept = int(((~flagged) & clean).sum())

    det_precision = float(tp_detect / n_flagged) if n_flagged > 0 else 0.0
    det_recall = float(tp_detect / n_poisoned) if n_poisoned > 0 else 0.0
    if det_precision + det_recall > 0:
        det_f1 = float(2.0 * det_precision * det_recall / (det_precision + det_recall))
    else:
        det_f1 = 0.0

    frac_clean_removed = float(fp_clean_removed / n_clean_train) if n_clean_train > 0 else 0.0
    frac_train_removed = float(n_flagged / n_train) if n_train > 0 else 0.0

    return {
        "n_train": n_train,
        "n_poisoned": n_poisoned,
        "n_clean_train": n_clean_train,
        "n_flagged": n_flagged,
        "n_true_detected": tp_detect,
        "n_clean_removed": fp_clean_removed,
        "n_missed_poison": fn_missed_poison,
        "n_clean_kept": tn_clean_kept,
        "frac_clean_removed": frac_clean_removed,
        "frac_train_removed": frac_train_removed,
        "detection_precision": det_precision,
        "detection_recall": det_recall,
        "detection_f1": det_f1,
    }


def compute_recovery_ratio(
    defended_metric: float,
    poisoned_metric: float,
    clean_metric: float,
) -> float:
    """Compute performance recovery ratio: (defended - poisoned) / (clean - poisoned).

    Retains negative values (when defense hurts performance) and returns np.nan when
    clean_metric == poisoned_metric (denominator is zero).
    """
    denom = float(clean_metric - poisoned_metric)
    if abs(denom) < 1e-12:
        return float("nan")
    return float((defended_metric - poisoned_metric) / denom)
