import json
from pathlib import Path
import numpy as np


class PCA:
    def __init__(self, n_components: int | float | None = None):
        if n_components is not None:
            if isinstance(n_components, float) and not (0.0 < n_components < 1.0):
                raise ValueError("Jika float, n_components harus berada dalam rentang (0.0, 1.0)")
            elif isinstance(n_components, int) and n_components < 1:
                raise ValueError("Jika int, n_components harus >= 1")

        self.n_components = n_components

        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.cumulative_explained_variance_ratio_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.n_components_: int | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray | list) -> "PCA":
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError(f"X harus berupa matriks 2D, didapat shape {X_arr.shape}")

        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        self.mean_ = np.mean(X_arr, axis=0)
        X_centered = X_arr - self.mean_

        cov_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        eigenvalues = np.maximum(eigenvalues, 0.0)
        total_variance = np.sum(eigenvalues)

        if total_variance > 0:
            variance_ratio = eigenvalues / total_variance
        else:
            variance_ratio = np.zeros_like(eigenvalues)

        self.explained_variance_ = eigenvalues
        self.explained_variance_ratio_ = variance_ratio
        self.cumulative_explained_variance_ratio_ = np.cumsum(variance_ratio)

        if self.n_components is None:
            k = n_features
        elif isinstance(self.n_components, int):
            k = min(self.n_components, n_features)
        else:  
            target_variance = float(self.n_components)
            k = int(np.searchsorted(self.cumulative_explained_variance_ratio_, target_variance) + 1)
            k = min(k, n_features)

        self.n_components_ = k

        self.components_ = eigenvectors[:, :k].T

        return self

    def transform(self, X: np.ndarray | list) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCA belum di-fit. Panggil .fit() terlebih dahulu.")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Jumlah fitur masukan ({X_arr.shape[1]}) tidak cocok dengan data fit ({self.n_features_in_})"
            )

        X_centered = X_arr - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X: np.ndarray | list) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_projected: np.ndarray | list) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCA belum di-fit. Panggil .fit() terlebih dahulu.")

        Z = np.asarray(X_projected, dtype=float)
        if Z.shape[1] != self.n_components_:
            raise ValueError(
                f"Jumlah komponen ({Z.shape[1]}) tidak sesuai dengan n_components_ ({self.n_components_})"
            )

        return np.dot(Z, self.components_) + self.mean_

    def get_params(self) -> dict:
        return {"n_components": self.n_components}

    def save(self, path: str | Path) -> Path:
        if self.components_ is None:
            raise RuntimeError("PCA belum di-fit.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "params": self.get_params(),
            "n_components_": self.n_components_,
            "n_features_in_": self.n_features_in_,
            "mean_": self.mean_.tolist(),
            "components_": self.components_.tolist(),
            "explained_variance_": self.explained_variance_.tolist(),
            "explained_variance_ratio_": self.explained_variance_ratio_.tolist(),
            "cumulative_explained_variance_ratio_": self.cumulative_explained_variance_ratio_.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PCA":
        """Muat model PCA dari berkas JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(**payload["params"])
        model.n_components_ = payload["n_components_"]
        model.n_features_in_ = payload["n_features_in_"]
        model.mean_ = np.array(payload["mean_"], dtype=float)
        model.components_ = np.array(payload["components_"], dtype=float)
        model.explained_variance_ = np.array(payload["explained_variance_"], dtype=float)
        model.explained_variance_ratio_ = np.array(payload["explained_variance_ratio_"], dtype=float)
        model.cumulative_explained_variance_ratio_ = np.array(
            payload["cumulative_explained_variance_ratio_"], dtype=float
        )
        return model
