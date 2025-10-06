# PyTorch soft-label classifier integrated with the existing MLPipeline.
# This estimator supports training with soft labels (probability distributions)
# and plugs into scikit-learn style pipelines used by xrd-analysis.

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore
    _TORCH_IMPORT_ERROR = e
else:
    _TORCH_IMPORT_ERROR = None


class _MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _ensure_torch():
    if torch is None:
        raise ImportError(
            "PyTorch is required for TorchSoftLabelClassifier. Please install it with `pip install torch --index-url https://download.pytorch.org/whl/cpu` (or the CUDA variant)."
        ) from _TORCH_IMPORT_ERROR


def _to_numpy_array_list(val) -> np.ndarray:
    """Converts list-like cell values to a 1D numpy array. Handles strings like "[0.1, 0.9]"."""
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
        except Exception:
            raise ValueError(f"Cannot parse string soft label '{val}' into a list.")
        val = parsed
    arr = np.asarray(val, dtype=float)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


def _extract_feature_matrix(
    X: Union[pd.DataFrame, np.ndarray],
    feature_column: Optional[Union[str, Sequence[str]]],
) -> np.ndarray:
    """Builds a numeric 2D feature matrix from X.

    - If feature_column is a string and present in X (DataFrame), it expects each entry
      to be a 1D array/list and stacks them to (n, d).
    - If feature_column is a list of column names, it flattens/concats them in order.
    - Else, if X is already a numeric DataFrame/ndarray, it uses its values.
    """
    if isinstance(X, pd.DataFrame):
        if isinstance(feature_column, str) and feature_column in X.columns:
            items = X[feature_column].tolist()
            mats = [np.asarray(z, dtype=float).ravel() for z in items]
            return np.vstack(mats)
        elif isinstance(feature_column, (list, tuple)):
            rows: List[np.ndarray] = []
            for _, row in X.iterrows():
                parts: List[np.ndarray] = []
                for col in feature_column:
                    val = row[col]
                    if isinstance(val, (list, np.ndarray)):
                        parts.append(np.asarray(val, dtype=float).ravel())
                    else:
                        parts.append(np.asarray([val], dtype=float))
                rows.append(np.concatenate(parts, axis=0))
            return np.vstack(rows)
        else:
            # try to use numeric columns directly
            return np.asarray(X.values, dtype=float)
    else:
        # X is ndarray-like
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr


class TorchSoftLabelClassifier(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible PyTorch classifier that supports training with
    soft labels (probability distributions).

    Integration strategy with current MLPipeline:
    - Keep wrangle=True, preprocess=False when training this estimator, so X is a DataFrame
      that still contains the array feature column (e.g., 'radial_profile_data') and, if used,
      a soft label column (e.g., 'cancer_status_soft').
    - Provide:
        - feature_column: name or names to extract and flatten as features.
        - soft_label_column: optional column containing per-sample probability vectors.
          If not provided, the estimator will look at 'y' passed by the pipeline; if 'y' has shape
          (n, C) or is float-valued in (0,1) and rows sum approximately to 1, it will be treated as
          soft labels; otherwise it will assume hard labels.

    Parameters
    ----------
    feature_column: str | list[str] | None
        Source feature column(s) in X to flatten into the input vector. If None,
        X is assumed already numeric.
    soft_label_column: str | None
        Name of the column in X with soft labels (each cell a list/array of length C).
    n_classes: int | None
        Number of classes; if None, inferred from soft labels or y.
    hidden_dims: list[int]
        Hidden layer sizes for the MLP.
    dropout: float
        Dropout probability.
    lr: float
        Learning rate.
    weight_decay: float
        L2 weight decay.
    epochs: int
        Number of training epochs.
    batch_size: int
        Batch size.
    device: str
        'cpu' or 'cuda'.
    seed: int
        Random seed.
    verbose: bool
        Print training progress.
    """

    def __init__(
        self,
        feature_column: Optional[Union[str, Sequence[str]]] = "radial_profile_data",
        soft_label_column: Optional[str] = None,
        n_classes: Optional[int] = None,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 50,
        batch_size: int = 64,
        device: str = "cpu",
        seed: int = 42,
        verbose: bool = False,
    ):
        self.feature_column = feature_column
        self.soft_label_column = soft_label_column
        self.n_classes = n_classes
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.device = device
        self.seed = int(seed)
        self.verbose = bool(verbose)

        # Runtime attributes
        self.model_: Optional[_MLP] = None
        self.classes_: Optional[np.ndarray] = None
        self.input_dim_: Optional[int] = None
        self.feature_names_in_: Optional[List[str]] = None

    # ------------------------
    # sklearn API
    # ------------------------
    def fit(
        self,
        X,
        y=None,
        sample_weight: Optional[Union[np.ndarray, Sequence[float]]] = None,
    ):
        _ensure_torch()
        rng = np.random.RandomState(self.seed)

        # Extract soft labels from X if available
        y_soft: Optional[np.ndarray] = None
        if (
            isinstance(X, pd.DataFrame)
            and self.soft_label_column
            and self.soft_label_column in X.columns
        ):
            raw = X[self.soft_label_column].tolist()
            y_soft = np.vstack([_to_numpy_array_list(v) for v in raw])
            # If given as single-prob (binary), expand to 2-class [1-p, p]
            if y_soft.ndim == 1 or (y_soft.ndim == 2 and y_soft.shape[1] == 1):
                y_soft = y_soft.reshape(-1)
                y_soft = np.stack([1.0 - y_soft, y_soft], axis=1)
            # Do not leak soft labels into feature matrix
            X = X.drop(columns=[self.soft_label_column])

        # If still no soft labels, try to infer from y
        if y_soft is None and y is not None:
            y_arr = np.asarray(y)
            if y_arr.ndim == 1:
                # Interpret as single-probability for positive class (binary case)
                y_prob = y_arr.astype(float)
                if np.nanmin(y_prob) >= 0.0 and np.nanmax(y_prob) <= 1.0:
                    y_soft = np.stack([1.0 - y_prob, y_prob], axis=1).astype(np.float32)
            elif y_arr.ndim == 2 and y_arr.shape[1] >= 2:  # shape (n, C)
                # Treat as soft labels if rows sum ~ 1 or entries in [0,1]
                row_sums = y_arr.sum(axis=1)
                if np.allclose(row_sums, 1.0, atol=1e-3) or (
                    np.nanmin(y_arr) >= 0.0 and np.nanmax(y_arr) <= 1.0
                ):
                    y_soft = y_arr.astype(np.float32)

        # Determine class count
        n_classes = int(self.n_classes) if self.n_classes is not None else None
        if n_classes is None:
            if y_soft is not None:
                n_classes = int(y_soft.shape[1])
            elif y is not None:
                y_arr = np.asarray(y)
                if y_arr.ndim == 1:
                    n_classes = int(len(np.unique(y_arr)))
        if n_classes is None:
            raise ValueError(
                "Cannot infer number of classes; specify n_classes or provide y/y_soft."
            )

        # Build features
        X_mat = _extract_feature_matrix(X, self.feature_column).astype(np.float32)
        self.input_dim_ = int(X_mat.shape[1])
        self.feature_names_in_ = (
            list(X.columns) if isinstance(X, pd.DataFrame) else None
        )

        # Build targets
        if y_soft is not None:
            Y = y_soft.astype(np.float32)
            # If rows don't perfectly sum to 1, renormalize
            s = Y.sum(axis=1, keepdims=True)
            s[s == 0] = 1.0
            Y = Y / s
        else:
            if y is None:
                raise ValueError(
                    "fit requires y when soft_label_column is not provided and y_soft not inferred."
                )
            y_arr = np.asarray(y)
            if y_arr.ndim != 1:
                raise ValueError("Hard labels y must be 1D when not using soft labels.")
            # Convert to 0..C-1
            uniq = np.unique(y_arr)
            mapping = {val: i for i, val in enumerate(sorted(uniq))}
            Y = np.array([mapping[v] for v in y_arr], dtype=np.int64)

        # Set classes_
        if y_soft is not None:
            self.classes_ = np.arange(n_classes)
        else:
            # classes from mapping order 0..C-1
            self.classes_ = np.arange(n_classes)

        # Torch tensors
        device = torch.device(
            self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu"
        )
        X_t = torch.from_numpy(X_mat)
        if y_soft is not None:
            Y_t = torch.from_numpy(Y)  # float32, shape (n, C)
        else:
            Y_t = torch.from_numpy(Y)  # int64, shape (n,)

        # Sample weights
        W_t: Optional[torch.Tensor] = None
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float32)
            if w.ndim != 1 or w.shape[0] != X_mat.shape[0]:
                raise ValueError("sample_weight must be 1D and match number of samples")
            W_t = torch.from_numpy(w)

        ds = TensorDataset(X_t, Y_t) if W_t is None else TensorDataset(X_t, Y_t, W_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        # Model
        torch.manual_seed(self.seed)
        model = _MLP(
            self.input_dim_,
            n_classes,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )
        model.to(device)
        opt = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        def loss_soft(
            logits: torch.Tensor,
            targets: torch.Tensor,
            weights: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            # targets: (B, C) soft labels
            log_probs = F.log_softmax(logits, dim=1)
            per_sample = -(targets * log_probs).sum(dim=1)
            if weights is not None:
                per_sample = per_sample * weights
                return per_sample.sum() / (weights.sum() + 1e-8)
            return per_sample.mean()

        def loss_hard(
            logits: torch.Tensor,
            targets: torch.Tensor,
            weights: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            # targets: (B,) int64
            log_probs = F.log_softmax(logits, dim=1)
            # one-hot
            oh = F.one_hot(targets, num_classes=n_classes).float()
            per_sample = -(oh * log_probs).sum(dim=1)
            if weights is not None:
                per_sample = per_sample * weights
                return per_sample.sum() / (weights.sum() + 1e-8)
            return per_sample.mean()

        # Train loop
        model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            n_batches = 0
            for batch in loader:
                if W_t is None:
                    xb, yb = batch
                    wb = None
                else:
                    xb, yb, wb = batch
                xb = xb.to(device)
                yb = yb.to(device)
                wb = wb.to(device) if wb is not None else None

                opt.zero_grad()
                logits = model(xb)
                if y_soft is not None:
                    loss_value = loss_soft(logits, yb, wb)
                else:
                    loss_value = loss_hard(logits, yb, wb)
                loss_value.backward()
                opt.step()

                total_loss += float(loss_value.detach().cpu().item())
                n_batches += 1

            if self.verbose and (
                epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1
            ):
                avg = total_loss / max(1, n_batches)
                print(
                    f"[TorchSoftLabelClassifier] epoch {epoch+1}/{self.epochs} loss={avg:.4f}"
                )

        self.model_ = model.eval()
        return self

    def _ensure_fitted(self):
        if self.model_ is None:
            raise RuntimeError("Estimator is not fitted. Call fit() first.")

    def _prepare_features(self, X) -> np.ndarray:
        X_mat = _extract_feature_matrix(X, self.feature_column).astype(np.float32)
        return X_mat

    def predict_proba(self, X):
        _ensure_torch()
        self._ensure_fitted()
        device = next(self.model_.parameters()).device  # type: ignore
        X_mat = self._prepare_features(X)
        with torch.no_grad():
            xb = torch.from_numpy(X_mat).to(device)
            logits = self.model_(xb)  # type: ignore
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        # Map to classes_
        classes = (
            self.classes_ if self.classes_ is not None else np.arange(proba.shape[1])
        )
        return classes[idx]
