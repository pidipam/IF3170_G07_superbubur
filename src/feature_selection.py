import json
from pathlib import Path
import numpy as np


def compute_correlation_matrix(X: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    return np.corrcoef(X_arr, rowvar=False)


def find_collinear_features(
    X: np.ndarray, feature_names: list[str] | None = None, threshold: float = 0.85
) -> list[tuple[str | int, str | int, float]]:

    corr = compute_correlation_matrix(X)
    n_features = corr.shape[0]
    names = feature_names if feature_names is not None else list(range(n_features))

    collinear_pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            val = corr[i, j]
            if np.isnan(val):
                continue
            if abs(val) >= threshold:
                collinear_pairs.append((names[i], names[j], float(val)))

    return collinear_pairs


def anova_f_score(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y).ravel()

    n_samples, n_features = X_arr.shape
    classes, counts = np.unique(y_arr, return_counts=True)
    n_classes = len(classes)

    if n_classes < 2:
        return np.zeros(n_features, dtype=float)

    grand_mean = np.mean(X_arr, axis=0)

    ss_between = np.zeros(n_features, dtype=float)
    ss_within = np.zeros(n_features, dtype=float)

    for c, count in zip(classes, counts):
        mask = y_arr == c
        Xc = X_arr[mask]
        mean_c = np.mean(Xc, axis=0)
        ss_between += count * ((mean_c - grand_mean) ** 2)
        ss_within += np.sum((Xc - mean_c) ** 2, axis=0)

    df_between = n_classes - 1
    df_within = n_samples - n_classes

    ms_between = ss_between / max(df_between, 1)
    ms_within = ss_within / max(df_within, 1)

    safe_ms_within = np.where(ms_within == 0.0, 1e-12, ms_within)
    f_scores = ms_between / safe_ms_within
    return np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)


def point_biserial_corr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    y_arr = (np.asarray(y).ravel() > 0).astype(int)

    n_samples, n_features = X_arr.shape
    mask_1 = y_arr == 1
    mask_0 = ~mask_1

    n1 = int(np.sum(mask_1))
    n0 = n_samples - n1

    if n0 == 0 or n1 == 0:
        return np.zeros(n_features, dtype=float)

    mean_1 = np.mean(X_arr[mask_1], axis=0)
    mean_0 = np.mean(X_arr[mask_0], axis=0)
    total_std = np.std(X_arr, axis=0, ddof=0)

    safe_std = np.where(total_std == 0.0, 1e-12, total_std)
    p = n1 / n_samples
    r_pb = ((mean_1 - mean_0) / safe_std) * np.sqrt(p * (1.0 - p))
    return np.nan_to_num(r_pb, nan=0.0)


def mutual_information_score(X: np.ndarray, y: np.ndarray, n_bins: int = 10) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y).ravel()

    n_samples, n_features = X_arr.shape
    classes, y_counts = np.unique(y_arr, return_counts=True)
    p_y = y_counts / n_samples
    h_y = -np.sum(p_y * np.log2(p_y + 1e-12))

    mi_scores = np.zeros(n_features, dtype=float)

    for j in range(n_features):
        col = X_arr[:, j]
        unique_vals = np.unique(col)
        if len(unique_vals) <= n_bins:
            binned = col
        else:
            bins = np.linspace(np.min(col), np.max(col), n_bins + 1)
            binned = np.digitize(col, bins[:-1])

        bin_vals, bin_counts = np.unique(binned, return_counts=True)
        h_y_given_x = 0.0

        for b_val, b_cnt in zip(bin_vals, bin_counts):
            sub_y = y_arr[binned == b_val]
            _, sub_y_counts = np.unique(sub_y, return_counts=True)
            p_sub = sub_y_counts / b_cnt
            h_sub = -np.sum(p_sub * np.log2(p_sub + 1e-12))
            h_y_given_x += (b_cnt / n_samples) * h_sub

        mi = h_y - h_y_given_x
        mi_scores[j] = max(0.0, float(mi))

    return mi_scores


def chi2_contingency_score(X_cat: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X_cat, dtype=float)
    y_arr = np.asarray(y).ravel()
    classes = np.unique(y_arr)

    n_samples, n_features = X_arr.shape
    scores = np.zeros(n_features, dtype=float)

    for j in range(n_features):
        col = X_arr[:, j]
        categories = np.unique(col)
        contingency = np.zeros((len(categories), len(classes)), dtype=float)

        for r_idx, cat in enumerate(categories):
            for c_idx, cls in enumerate(classes):
                contingency[r_idx, c_idx] = np.sum((col == cat) & (y_arr == cls))

        row_sums = contingency.sum(axis=1, keepdims=True)
        col_sums = contingency.sum(axis=0, keepdims=True)
        expected = (row_sums @ col_sums) / max(n_samples, 1)

        safe_expected = np.where(expected == 0.0, 1e-12, expected)
        chi2_val = np.sum(((contingency - expected) ** 2) / safe_expected)
        scores[j] = max(0.0, float(chi2_val))

    return scores


class FeatureSelector:
    def __init__(self, k: int | None = None, method: str = "anova", threshold: float | None = None):
        if method not in ("anova", "mutual_info", "correlation"):
            raise ValueError("method harus 'anova', 'mutual_info', atau 'correlation'")
        if k is not None and k < 1:
            raise ValueError("k harus >= 1")

        self.k = k
        self.method = method
        self.threshold = threshold

        self.scores_: np.ndarray | None = None
        self.ranking_: np.ndarray | None = None
        self.selected_indices_: np.ndarray | None = None
        self.selected_features_: list[str] | None = None
        self.feature_names_: list[str] | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> "FeatureSelector":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y).ravel()
        n_samples, n_features = X_arr.shape

        self.n_features_in_ = n_features
        self.feature_names_ = (
            list(feature_names) if feature_names is not None else [f"f_{i}" for i in range(n_features)]
        )

        if len(self.feature_names_) != n_features:
            raise ValueError("Panjang feature_names harus sama dengan jumlah kolom X")

        if self.method == "anova":
            self.scores_ = anova_f_score(X_arr, y_arr)
        elif self.method == "mutual_info":
            self.scores_ = mutual_information_score(X_arr, y_arr)
        elif self.method == "correlation":
            self.scores_ = np.abs(point_biserial_corr(X_arr, y_arr))

        self.ranking_ = np.argsort(self.scores_)[::-1]

        if self.k is not None:
            k = min(self.k, n_features)
            self.selected_indices_ = np.sort(self.ranking_[:k])
        elif self.threshold is not None:
            self.selected_indices_ = np.where(self.scores_ >= self.threshold)[0]
            if len(self.selected_indices_) == 0:
                self.selected_indices_ = np.array([self.ranking_[0]])
        else:
            self.selected_indices_ = np.arange(n_features)

        self.selected_features_ = [self.feature_names_[i] for i in self.selected_indices_]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_indices_ is None:
            raise RuntimeError("FeatureSelector belum di-fit.")
        X_arr = np.asarray(X, dtype=float)
        return X_arr[:, self.selected_indices_]

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None
    ) -> np.ndarray:
        return self.fit(X, y, feature_names=feature_names).transform(X)

    def get_feature_scores(self) -> list[dict]:
        if self.scores_ is None or self.ranking_ is None:
            raise RuntimeError("FeatureSelector belum difit.")

        summary = []
        for rank, idx in enumerate(self.ranking_):
            summary.append(
                {
                    "rank": rank + 1,
                    "index": int(idx),
                    "feature": self.feature_names_[idx],
                    "score": float(self.scores_[idx]),
                    "selected": bool(idx in self.selected_indices_),
                }
            )
        return summary

    def save(self, path: str | Path) -> Path:
        if self.selected_indices_ is None:
            raise RuntimeError("FeatureSelector belum difit.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "params": {"k": self.k, "method": self.method, "threshold": self.threshold},
            "selected_features_": self.selected_features_,
            "selected_indices_": self.selected_indices_.tolist(),
            "scores_": self.scores_.tolist(),
            "ranking_": self.ranking_.tolist(),
            "feature_names_": self.feature_names_,
            "n_features_in_": self.n_features_in_,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSelector":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        fs = cls(**payload["params"])
        fs.selected_features_ = payload["selected_features_"]
        fs.selected_indices_ = np.array(payload["selected_indices_"], dtype=int)
        fs.scores_ = np.array(payload["scores_"], dtype=float)
        fs.ranking_ = np.array(payload["ranking_"], dtype=int)
        fs.feature_names_ = payload["feature_names_"]
        fs.n_features_in_ = payload["n_features_in_"]
        return fs
