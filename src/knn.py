from pathlib import Path
import numpy as np

_EPS = 1e-12

class KNN:
    def __init__(
        self,
        k: int = 5,
        p: float = 2.0,
        weights: str = "uniform",
        categorical_features=None,
        batch_size: int = 512,
    ):
        if int(k) < 1:
            raise ValueError("k harus >= 1")
        if p < 1:
            raise ValueError("p harus >= 1")
        if weights not in ("uniform", "distance"):
            raise ValueError("weights harus 'uniform' atau 'distance'")
        self.k = int(k)
        self.p = float(p)
        self.weights = weights
        self.categorical_features = (
            None if categorical_features is None else [int(i) for i in categorical_features]
        )
        self.batch_size = int(batch_size)

        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None
        self._y_idx = None
        self._num_idx = None
        self._cat_idx = None

    # Fit
    def fit(self, X, y) -> "KNN":
        """Menyimpan data latih (lazy learner: tidak ada proses belajar)."""
        X = self._check_X(X)
        y = np.asarray(y).ravel()
        if len(X) != len(y):
            raise ValueError("X dan y harus punya jumlah baris yang sama")
        if self.k > len(X):
            raise ValueError("k tidak boleh lebih besar dari jumlah data latih")

        self.X_train_ = X
        self.y_train_ = y
        self.classes_, self._y_idx = np.unique(y, return_inverse=True)
        self._split_feature_types(X.shape[1])
        return self

    def _split_feature_types(self, n_features: int) -> None:
        cat = np.array(sorted(set(self.categorical_features or [])), dtype=int)
        if len(cat) and (cat.min() < 0 or cat.max() >= n_features):
            raise ValueError("categorical_features berisi indeks di luar jumlah kolom")
        self._cat_idx = cat
        self._num_idx = np.setdiff1d(np.arange(n_features), cat)

    # Distances
    def _distance_batch(self, Q: np.ndarray) -> np.ndarray:
        """Jarak Minkowski gabungan antara tiap baris Q dan seluruh data latih."""
        T = self.X_train_
        acc = np.zeros((len(Q), len(T)), dtype=float)

        if len(self._num_idx):
            Qn, Tn = Q[:, self._num_idx], T[:, self._num_idx]
            if self.p == 2.0:
                # ||q - t||^2 = ||q||^2 + ||t||^2 - 2 q.t  
                acc += (Qn**2).sum(1)[:, None] + (Tn**2).sum(1)[None, :] - 2.0 * (Qn @ Tn.T)
                np.maximum(acc, 0.0, out=acc)  # buang galat pembulatan negatif
            else:
                buf = np.empty_like(acc)  # buffer dipakai ulang agar tidak alokasi array besar per fitur
                for j in range(Qn.shape[1]):
                    np.subtract(Qn[:, j][:, None], Tn[:, j][None, :], out=buf)
                    np.abs(buf, out=buf)
                    if self.p != 1.0:
                        np.power(buf, self.p, out=buf)
                    acc += buf

        for j in self._cat_idx:  # atribut simbolik: 1 jika beda, 0 jika sama
            acc += Q[:, j][:, None] != T[:, j][None, :]

        return np.sqrt(acc) if self.p == 2.0 else acc ** (1.0 / self.p)

    def kneighbors(self, X, k: int | None = None, return_distance: bool = True):
        self._check_fitted()
        k = self.k if k is None else int(k)
        if not 1 <= k <= len(self.X_train_):
            raise ValueError("k di luar rentang 1..jumlah data latih")
        X = self._check_X(X)
        if X.shape[1] != self.X_train_.shape[1]:
            raise ValueError("jumlah kolom X berbeda dengan data latih")

        all_idx, all_dist = [], []
        for start in range(0, len(X), self.batch_size):
            D = self._distance_batch(X[start : start + self.batch_size])
            idx = np.argsort(D, axis=1, kind="stable")[:, :k]
            all_idx.append(idx)
            if return_distance:
                all_dist.append(np.take_along_axis(D, idx, axis=1))

        idx = np.vstack(all_idx)
        return (np.vstack(all_dist), idx) if return_distance else idx

    # Prediction
    def vote(self, dist: np.ndarray, idx: np.ndarray, k: int, weights: str | None = None):
        weights = self.weights if weights is None else weights
        labels = self._y_idx[idx[:, :k]]  # (m, k) indeks kelas
        if weights == "uniform":
            w = np.ones_like(dist[:, :k])
        else:
            w = 1.0 / (dist[:, :k] + _EPS)

        scores = np.zeros((len(labels), len(self.classes_)))
        for c in range(len(self.classes_)):
            scores[:, c] = (w * (labels == c)).sum(axis=1)
        proba = scores / scores.sum(axis=1, keepdims=True)

        # Kalau ada beberapa yang seri pilih neighbor terdekat
        best = scores.max(axis=1, keepdims=True)
        tied = np.isclose(scores, best, rtol=1e-9, atol=0.0)
        pred = scores.argmax(axis=1)
        needs_break = tied.sum(axis=1) > 1
        if needs_break.any():
            rows = np.where(needs_break)[0]
            for r in rows:
                for neighbor_class in labels[r]:  # sudah terurut dari terdekat
                    if tied[r, neighbor_class]:
                        pred[r] = neighbor_class
                        break
        return proba, self.classes_[pred]

    def predict_proba(self, X) -> np.ndarray:
        dist, idx = self.kneighbors(X)
        return self.vote(dist, idx, self.k)[0]

    def predict(self, X) -> np.ndarray:
        dist, idx = self.kneighbors(X)
        return self.vote(dist, idx, self.k)[1]

    def predict_k_range(self, X, ks) -> dict:
        ks = sorted({int(k) for k in ks})
        dist, idx = self.kneighbors(X, k=ks[-1])
        return {k: self.vote(dist, idx, k)[1] for k in ks}

    def score(self, X, y) -> float:
        """Akurasi."""
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

    # Persistence
    def save(self, path) -> Path:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X_train=self.X_train_,
            y_train=self.y_train_,
            k=self.k,
            p=self.p,
            weights=self.weights,
            categorical_features=np.array(self.categorical_features or [], dtype=int),
            has_categorical=self.categorical_features is not None,
            batch_size=self.batch_size,
        )
        return path

    @classmethod
    def load(cls, path) -> "KNN":
        with np.load(Path(path), allow_pickle=False) as f:
            cat = f["categorical_features"].tolist() if bool(f["has_categorical"]) else None
            model = cls(
                k=int(f["k"]),
                p=float(f["p"]),
                weights=str(f["weights"]),
                categorical_features=cat,
                batch_size=int(f["batch_size"]),
            )
            return model.fit(f["X_train"], f["y_train"])

    # Helpers
    def get_params(self) -> dict:
        return {
            "k": self.k,
            "p": self.p,
            "weights": self.weights,
            "categorical_features": self.categorical_features,
            "batch_size": self.batch_size,
        }

    @staticmethod
    def _check_X(X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X harus 2 dimensi (jumlah_data, jumlah_fitur)")
        if not np.isfinite(X).all():
            raise ValueError("X mengandung NaN/inf -- lakukan imputasi di tahap preprocessing")
        return X

    def _check_fitted(self) -> None:
        if self.X_train_ is None:
            raise RuntimeError("Model belum di-fit. Panggil fit(X, y) terlebih dahulu.")

    def __repr__(self) -> str:
        return (
            f"KNN(k={self.k}, p={self.p:g}, weights='{self.weights}', "
            f"categorical_features={self.categorical_features})"
        )