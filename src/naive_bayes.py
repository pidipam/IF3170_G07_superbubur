import json
from pathlib import Path
import numpy as np


class DiscreteNaiveBayes:
    def __init__(self, alpha: float = 1.0, n_bins: int | None = 10, binning: str = "quantile"):
        if alpha < 0:
            raise ValueError("alpha harus >= 0")
        if n_bins is not None and n_bins < 2:
            raise ValueError("n_bins harus >= 2 jika dispesifikasikan")
        if binning not in ("quantile", "uniform"):
            raise ValueError("binning harus 'quantile' atau 'uniform'")

        self.alpha = float(alpha)
        self.n_bins = n_bins
        self.binning = binning

        self.classes_: np.ndarray | None = None
        self.class_priors_: np.ndarray | None = None
        self.class_counts_: np.ndarray | None = None
        self.feature_tables_: dict[any, list[np.ndarray]] = {}  
        self.bin_edges_: list[np.ndarray] = []
        self.n_features_in_: int | None = None

    def _discretize_fit(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        self.bin_edges_ = []
        X_binned = np.zeros((n_samples, n_features), dtype=int)

        for j in range(n_features):
            col = X[:, j]
            unique_vals = np.unique(col)

            if self.n_bins is None or len(unique_vals) <= self.n_bins:
                edges = np.sort(unique_vals)
                self.bin_edges_.append(edges)
                X_binned[:, j] = np.searchsorted(edges, col)
            else:
                if self.binning == "quantile":
                    quantiles = np.linspace(0, 100, self.n_bins + 1)
                    edges = np.unique(np.percentile(col, quantiles))
                else:
                    edges = np.linspace(np.min(col), np.max(col), self.n_bins + 1)

                cut_edges = edges[1:-1]
                self.bin_edges_.append(cut_edges)
                X_binned[:, j] = np.digitize(col, cut_edges)

        return X_binned

    def _discretize_transform(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        X_binned = np.zeros((n_samples, n_features), dtype=int)

        for j in range(n_features):
            edges = self.bin_edges_[j]
            max_idx = len(self.feature_tables_[self.classes_[0]][j]) - 1

            if len(edges) == 0:
                X_binned[:, j] = 0
            elif np.all(edges[:-1] <= edges[1:]):  
                if len(edges) <= (self.n_bins or len(edges)):
                    idx = np.searchsorted(edges, X[:, j])
                    idx = np.clip(idx, 0, max_idx)
                    X_binned[:, j] = idx
                else:
                    X_binned[:, j] = np.clip(np.digitize(X[:, j], edges), 0, max_idx)
            else:
                X_binned[:, j] = np.clip(np.digitize(X[:, j], edges), 0, max_idx)

        return X_binned

    def fit(self, X: np.ndarray | list, y: np.ndarray | list) -> "DiscreteNaiveBayes":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y).ravel()

        if len(X_arr) != len(y_arr):
            raise ValueError(f"Jumlah sampel X ({len(X_arr)}) dan y ({len(y_arr)}) tidak sama")

        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        self.classes_, class_counts = np.unique(y_arr, return_counts=True)
        self.class_counts_ = class_counts.astype(float)
        self.class_priors_ = self.class_counts_ / n_samples

        X_binned = self._discretize_fit(X_arr)

        self.feature_tables_ = {}
        for c_idx, c in enumerate(self.classes_):
            Xc = X_binned[y_arr == c]
            n_c = len(Xc)
            self.feature_tables_[c] = []

            for j in range(n_features):
                edges = self.bin_edges_[j]
                K_j = len(edges) + 1 if len(edges) > 0 else 1

                counts = np.bincount(Xc[:, j], minlength=K_j)

                if self.alpha > 0:
                    probs = (counts + self.alpha) / (n_c + self.alpha * K_j)
                else:
                    safe_nc = max(n_c, 1)
                    probs = counts / safe_nc
                    probs = np.where(probs == 0, 1e-12, probs)

                self.feature_tables_[c].append(probs)

        return self

    def predict_log_proba(self, X: np.ndarray | list) -> np.ndarray:
        if not self.feature_tables_:
            raise RuntimeError("Model belum di-fit.")

        X_arr = np.asarray(X, dtype=float)
        n_samples, n_features = X_arr.shape
        X_binned = self._discretize_transform(X_arr)

        log_probs = np.zeros((n_samples, len(self.classes_)), dtype=float)
        for c_idx, c in enumerate(self.classes_):
            log_p = np.log(self.class_priors_[c_idx])
            for j in range(n_features):
                table_j = self.feature_tables_[c][j]
                bin_indices = X_binned[:, j]
                log_p += np.log(table_j[bin_indices])
            log_probs[:, c_idx] = log_p

        return log_probs

    def predict_proba(self, X: np.ndarray | list) -> np.ndarray:
        log_probs = self.predict_log_proba(X)
        max_log = np.max(log_probs, axis=1, keepdims=True)
        exp_p = np.exp(log_probs - max_log)
        return exp_p / np.sum(exp_p, axis=1, keepdims=True)

    def predict(self, X: np.ndarray | list, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        if len(self.classes_) == 2 and threshold != 0.5:
            pos_idx = 1 if self.classes_[1] == 1 else 0
            return np.where(proba[:, pos_idx] >= threshold, self.classes_[1], self.classes_[0])
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: np.ndarray | list, y: np.ndarray | list) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

    def get_params(self) -> dict:
        return {"alpha": self.alpha, "n_bins": self.n_bins, "binning": self.binning}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "discrete",
            "params": self.get_params(),
            "classes_": self.classes_.tolist(),
            "class_priors_": self.class_priors_.tolist(),
            "class_counts_": self.class_counts_.tolist(),
            "bin_edges_": [e.tolist() for e in self.bin_edges_],
            "feature_tables_": {
                str(k): [t.tolist() for t in tbls] for k, tbls in self.feature_tables_.items()
            },
            "n_features_in_": self.n_features_in_,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DiscreteNaiveBayes":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(**payload["params"])
        model.classes_ = np.array(payload["classes_"])
        model.class_priors_ = np.array(payload["class_priors_"], dtype=float)
        model.class_counts_ = np.array(payload["class_counts_"], dtype=float)
        model.bin_edges_ = [np.array(e, dtype=float) for e in payload["bin_edges_"]]
        model.feature_tables_ = {
            int(k) if k.lstrip("-").isdigit() else k: [np.array(t, dtype=float) for t in tbls]
            for k, tbls in payload["feature_tables_"].items()
        }
        model.n_features_in_ = int(payload["n_features_in_"])
        return model


class GaussianNaiveBayes:
    def __init__(self, var_smoothing: float = 1e-9, priors: np.ndarray | None = None):
        if var_smoothing <= 0:
            raise ValueError("var_smoothing harus bernilai positif > 0")

        self.var_smoothing = float(var_smoothing)
        self.priors = None if priors is None else np.asarray(priors, dtype=float)

        self.classes_: np.ndarray | None = None
        self.class_prior_: np.ndarray | None = None
        self.class_count_: np.ndarray | None = None
        self.theta_: np.ndarray | None = None
        self.var_: np.ndarray | None = None  
        self.epsilon_: float | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray | list, y: np.ndarray | list) -> "GaussianNaiveBayes":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y).ravel()

        if len(X_arr) != len(y_arr):
            raise ValueError("Jumlah sampel X dan y tidak sama")

        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        self.classes_, class_indices, class_counts = np.unique(
            y_arr, return_inverse=True, return_counts=True
        )
        n_classes = len(self.classes_)
        self.class_count_ = class_counts.astype(float)

        if self.priors is not None:
            self.class_prior_ = self.priors
        else:
            self.class_prior_ = self.class_count_ / n_samples

        global_variance = np.var(X_arr, axis=0)
        max_variance = np.max(global_variance) if len(global_variance) > 0 else 1.0
        self.epsilon_ = self.var_smoothing * max(max_variance, 1e-12)

        self.theta_ = np.zeros((n_classes, n_features), dtype=float)
        self.var_ = np.zeros((n_classes, n_features), dtype=float)

        for idx, c in enumerate(self.classes_):
            Xc = X_arr[y_arr == c]
            self.theta_[idx] = np.mean(Xc, axis=0)
            self.var_[idx] = np.var(Xc, axis=0) + self.epsilon_

        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        n_samples = X_arr.shape[0]
        n_classes = len(self.classes_)

        log_prior = np.log(self.class_prior_)
        joint_log_lik = np.zeros((n_samples, n_classes), dtype=float)

        for idx in range(n_classes):
            mean = self.theta_[idx]
            var = self.var_[idx]
            log_component = -0.5 * np.sum(
                np.log(2.0 * np.pi * var) + ((X_arr - mean) ** 2) / var,
                axis=1,
            )
            joint_log_lik[:, idx] = log_prior[idx] + log_component

        return joint_log_lik

    def predict_log_proba(self, X: np.ndarray | list) -> np.ndarray:
        joint_log = self._joint_log_likelihood(np.asarray(X, dtype=float))
        max_log = np.max(joint_log, axis=1, keepdims=True)
        log_prob_x = max_log + np.log(np.sum(np.exp(joint_log - max_log), axis=1, keepdims=True))
        return joint_log - log_prob_x

    def predict_proba(self, X: np.ndarray | list) -> np.ndarray:
        log_proba = self.predict_log_proba(X)
        proba = np.exp(log_proba)
        return proba / np.sum(proba, axis=1, keepdims=True)

    def predict(self, X: np.ndarray | list, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        if len(self.classes_) == 2 and threshold != 0.5:
            pos_idx = 1 if self.classes_[1] == 1 else 0
            return np.where(proba[:, pos_idx] >= threshold, self.classes_[1], self.classes_[0])
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: np.ndarray | list, y: np.ndarray | list) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

    def get_params(self) -> dict:
        return {
            "var_smoothing": self.var_smoothing,
            "priors": None if self.priors is None else self.priors.tolist(),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "gaussian",
            "params": self.get_params(),
            "classes_": self.classes_.tolist(),
            "class_prior_": self.class_prior_.tolist(),
            "class_count_": self.class_count_.tolist(),
            "theta_": self.theta_.tolist(),
            "var_": self.var_.tolist(),
            "epsilon_": float(self.epsilon_),
            "n_features_in_": self.n_features_in_,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "GaussianNaiveBayes":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(**payload["params"])
        model.classes_ = np.array(payload["classes_"])
        model.class_prior_ = np.array(payload["class_prior_"], dtype=float)
        model.class_count_ = np.array(payload["class_count_"], dtype=float)
        model.theta_ = np.array(payload["theta_"], dtype=float)
        model.var_ = np.array(payload["var_"], dtype=float)
        model.epsilon_ = float(payload["epsilon_"])
        model.n_features_in_ = int(payload["n_features_in_"])
        return model


class NaiveBayes:
    def __init__(
        self,
        method: str = "discrete",
        alpha: float = 1.0,
        n_bins: int | None = 10,
        var_smoothing: float = 1e-9,
        priors: np.ndarray | None = None,
    ):
        if method not in ("discrete", "gaussian"):
            raise ValueError("method harus 'discrete' atau 'gaussian'")

        self.method = method
        self.alpha = alpha
        self.n_bins = n_bins
        self.var_smoothing = var_smoothing
        self.priors = priors

        if method == "discrete":
            self.model_ = DiscreteNaiveBayes(alpha=alpha, n_bins=n_bins)
        else:
            self.model_ = GaussianNaiveBayes(var_smoothing=var_smoothing, priors=priors)

    def fit(self, X, y) -> "NaiveBayes":
        self.model_.fit(X, y)
        return self

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return self.model_.predict(X, threshold=threshold)

    def predict_proba(self, X) -> np.ndarray:
        return self.model_.predict_proba(X)

    def predict_log_proba(self, X) -> np.ndarray:
        return self.model_.predict_log_proba(X)

    def score(self, X, y) -> float:
        return self.model_.score(X, y)

    @property
    def classes_(self):
        return self.model_.classes_

    def get_params(self) -> dict:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "n_bins": self.n_bins,
            "var_smoothing": self.var_smoothing,
            "priors": None if self.priors is None else np.asarray(self.priors).tolist(),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": self.method,
            "params": self.get_params(),
            "submodel_path": str(path.with_suffix(".sub.json").name),
        }
        self.model_.save(path.with_suffix(".sub.json"))
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "NaiveBayes":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        method = payload["method"]
        submodel_path = path.parent / payload["submodel_path"]
        if method == "discrete":
            submodel = DiscreteNaiveBayes.load(submodel_path)
        else:
            submodel = GaussianNaiveBayes.load(submodel_path)

        nb = cls(**payload["params"])
        nb.model_ = submodel
        return nb
