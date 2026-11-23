import pickle
from pathlib import Path
import numpy as np
from scipy.optimize import Bounds, minimize


class SVM:
    """Binary Support Vector Machine (Soft-Margin) berbasis Dual QP (SciPy).

    Menyelesaikan masalah optimasi dual SVM:
        min_alpha  0.5 * alpha^T Q alpha - sum(alpha)
        s.t.       0 <= alpha_i <= C  dan  sum(alpha_i * y_i) = 0

    menggunakan `scipy.optimize.minimize` (L-BFGS-B + Augmented Lagrangian).

    Mendukung kernel:
    - 'linear' : K(x, x') = x . x'
    - 'poly'   : K(x, x') = (gamma * (x . x') + coef0) ** degree
    - 'gauss'  : K(x, x') = exp(-gamma * ||x - x'||^2)  (alias: 'rbf')
    """

    _VALID_KERNELS = ("linear", "poly", "gauss", "rbf")

    def __init__(
        self,
        kernel="linear",
        C=1.0,
        degree=3,
        coef0=1.0,  
        gamma="scale",  
        tol=1e-3,
        max_passes=3,
        random_state=42,
        max_iter=2000,
        cache_size=256.0,  
    ):
        if kernel not in self._VALID_KERNELS:
            raise ValueError(
                f"kernel harus salah satu dari {self._VALID_KERNELS}, bukan {kernel!r}."
            )
        if C <= 0:
            raise ValueError("C harus bernilai > 0.")
        if degree < 1:
            raise ValueError("degree harus bernilai >= 1.")
        if isinstance(gamma, str):
            if gamma not in ("scale", "auto"):
                raise ValueError("gamma string harus 'scale' atau 'auto'.")
        elif float(gamma) <= 0:
            raise ValueError("gamma numerik harus bernilai > 0.")
        if tol <= 0:
            raise ValueError("tol harus bernilai > 0.")
        if max_iter is not None and max_iter <= 0:
            raise ValueError("max_iter harus None atau bilangan bulat > 0.")
        if cache_size <= 0:
            raise ValueError("cache_size harus bernilai > 0.")

        self.kernel = kernel
        self.C = float(C)
        self.degree = degree
        self.coef0 = float(coef0)
        self.gamma = gamma
        self.tol = float(tol)
        self.max_passes = int(max_passes)
        self.random_state = random_state
        self.max_iter = None if max_iter is None else int(max_iter)
        self.cache_size = float(cache_size)

        self.w = None
        self.b = 0.0
        self.alphas = None
        self.sv_x = None
        self.sv_y = None
        self.support_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.n_iter_ = 0
        self._gamma_val = None

    @property
    def _is_gauss(self) -> bool:
        return self.kernel in ("gauss", "rbf")

    @staticmethod
    def _fast_power_inplace(arr, degree):
        """Pangkat integer cepat via perkalian in-place."""
        if isinstance(degree, (int, np.integer)) or (
            isinstance(degree, float) and degree.is_integer()
        ):
            d = int(degree)
            if d == 1:
                return arr
            if d == 2:
                arr *= arr
                return arr
            if d == 3:
                sq = arr * arr
                arr *= sq
                return arr
            result = np.ones_like(arr)
            base = arr
            while d > 0:
                if d & 1:
                    result *= base
                d >>= 1
                if d > 0:
                    base = base * base
            return result
        np.power(arr, degree, out=arr)
        return arr

    def _compute_kernel(self, X1, X2, X1_sq=None, X2_sq=None):
        if self.kernel == "linear":
            return X1 @ X2.T
        if self.kernel == "poly":
            K = X1 @ X2.T
            K *= self._gamma_val
            K += self.coef0
            return self._fast_power_inplace(K, self.degree)
        if self._is_gauss:
            if X1_sq is None:
                X1_sq = np.einsum("ij,ij->i", X1, X1)
            if X2_sq is None:
                X2_sq = np.einsum("ij,ij->i", X2, X2)
            sq_dists = X1 @ X2.T
            sq_dists *= -2.0
            sq_dists += X1_sq[:, None]
            sq_dists += X2_sq[None, :]
            np.maximum(sq_dists, 0.0, out=sq_dists)
            sq_dists *= -self._gamma_val
            np.exp(sq_dists, out=sq_dists)
            return sq_dists
        raise ValueError(f"kernel tidak valid: {self.kernel!r}.")

    def _matvec_K(self, X, ay, K_full=None, X_sq=None, batch_size=2048):
        """Hitung f0 = K(X, X) @ (alpha * y) secara efisien."""
        if self.kernel == "linear":
            return X @ (ay @ X)
        if K_full is not None:
            return K_full @ ay

        n = X.shape[0]
        nz = ay != 0.0
        if not np.any(nz):
            return np.zeros(n, dtype=np.float64)
        X_nz = X[nz]
        ay_nz = ay[nz]
        X_nz_sq = X_sq[nz] if X_sq is not None else None
        out = np.empty(n, dtype=np.float64)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            K_blk = self._compute_kernel(
                X[start:end],
                X_nz,
                X1_sq=(X_sq[start:end] if X_sq is not None else None),
                X2_sq=X_nz_sq,
            )
            out[start:end] = K_blk @ ay_nz
        return out

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X harus berupa array 2D non-kosong (n_samples, n_features).")
        if not np.isfinite(X).all():
            raise ValueError("X mengandung nilai NaN atau inf.")

        y_raw = np.asarray(y).ravel()
        if X.shape[0] != y_raw.shape[0]:
            raise ValueError(
                f"Jumlah sampel X ({X.shape[0]}) dan y ({y_raw.shape[0]}) tidak sama."
            )

        classes = np.unique(y_raw)
        if len(classes) != 2:
            raise ValueError(
                f"Binary SVM membutuhkan tepat 2 kelas unik, ditemukan {len(classes)}: {classes}."
            )

        self.classes_ = classes
        y_pm = np.where(y_raw == classes[1], 1.0, -1.0)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.w = None

        if self.gamma == "scale":
            var = float(X.var())
            self._gamma_val = 1.0 / (n_features * var) if var > 0.0 else 1.0
        elif self.gamma == "auto":
            self._gamma_val = 1.0 / n_features
        else:
            self._gamma_val = float(self.gamma)

        X_sq = np.einsum("ij,ij->i", X, X) if self._is_gauss else None
        cache_bytes = int(self.cache_size * 1024 * 1024)
        K_full = None
        if self.kernel != "linear" and (n_samples * n_samples * 8) <= cache_bytes:
            K_full = self._compute_kernel(X, X, X1_sq=X_sq, X2_sq=X_sq)

        C = self.C
        tol = self.tol
        max_total_iter = 3000 if self.max_iter is None else self.max_iter
        bounds = Bounds(0.0, 1.0)

        u = np.zeros(n_samples, dtype=np.float64)
        lam = 0.0  
        rho = 1.0
        eq_tol = 1e-5
        prev_eq = np.inf
        total_iter = 0

        for outer in range(25):
            remaining = max_total_iter - total_iter
            if remaining <= 0:
                break

            def fun_and_grad(u_vec):
                uy = u_vec * y_pm
                f0 = C * self._matvec_K(X, uy, K_full=K_full, X_sq=X_sq)
                eq = float(np.sum(uy))
                obj = (
                    0.5 * float(np.dot(uy, f0))
                    - float(np.sum(u_vec))
                    + lam * eq
                    + 0.5 * rho * eq * eq
                )
                grad = y_pm * (f0 + lam + rho * eq) - 1.0
                return obj, grad

            inner_maxiter = min(remaining, 250 if outer < 6 else 500)
            inner_gtol = min(tol, max(1e-6, tol * (0.5 ** min(outer, 5))))
            inner_ftol = 1e-10 if prev_eq > 1e-2 else 1e-13
            res = minimize(
                fun_and_grad,
                u,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={
                    "ftol": inner_ftol,
                    "gtol": inner_gtol,
                    "maxiter": inner_maxiter,
                },
            )
            u = np.clip(res.x, 0.0, 1.0)
            total_iter += int(res.nit)

            eq = float(np.dot(u, y_pm))
            lam += rho * eq

            if abs(eq) <= eq_tol:
                _, grad = fun_and_grad(u)
                proj_grad = np.where(
                    u <= 1e-12,
                    np.minimum(grad, 0.0),
                    np.where(u >= 1.0 - 1e-12, np.maximum(grad, 0.0), grad),
                )
                if float(np.max(np.abs(proj_grad))) <= tol:
                    break

            if abs(eq) > 0.25 * prev_eq:
                rho = min(rho * 5.0, 1e10)
            prev_eq = abs(eq)

        alphas = C * u

        f0 = self._matvec_K(X, alphas * y_pm, K_full=K_full, X_sq=X_sq)
        r_full = y_pm - f0
        free_mask = (u > 1e-5) & (u < 1.0 - 1e-5)
        if np.any(free_mask):
            b = float(np.mean(r_full[free_mask]))
        else:
            pos = y_pm > 0.0
            i_up = (pos & (u < 1.0 - 1e-12)) | ((~pos) & (u > 1e-12))
            i_low = ((~pos) & (u < 1.0 - 1e-12)) | (pos & (u > 1e-12))
            if np.any(i_up) and np.any(i_low):
                b = 0.5 * (float(np.max(r_full[i_up])) + float(np.min(r_full[i_low])))
            else:
                b = float(lam)

        sv_mask = u > 1e-6
        self.support_ = np.where(sv_mask)[0]
        self.alphas = alphas[sv_mask]
        self.sv_x = X[sv_mask]
        self.sv_y = y_pm[sv_mask]
        self.b = b
        self.n_iter_ = total_iter

        if self.kernel == "linear":
            self.w = (self.alphas * self.sv_y) @ self.sv_x

        return self

    def _check_fitted(self):
        if self.classes_ is None or self.sv_x is None:
            raise RuntimeError("Model belum di-fit. Panggil fit(X, y) terlebih dahulu.")

    def decision_function(self, X, batch_size=2048):
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X harus berupa array 2D (n_samples, n_features).")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Jumlah fitur X ({X.shape[1]}) tidak sesuai dengan data latih ({self.n_features_in_})."
            )

        if self.kernel == "linear":
            return X @ self.w + self.b

        if len(self.alphas) == 0:
            return np.full(X.shape[0], self.b, dtype=np.float64)

        ay = self.alphas * self.sv_y
        n = X.shape[0]
        if n <= batch_size:
            K_eval = self._compute_kernel(X, self.sv_x)
            return K_eval @ ay + self.b

        sv_sq = np.einsum("ij,ij->i", self.sv_x, self.sv_x) if self._is_gauss else None
        scores = np.empty(n, dtype=np.float64)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            K_blk = self._compute_kernel(X[start:end], self.sv_x, X2_sq=sv_sq)
            scores[start:end] = K_blk @ ay + self.b
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        y_true = np.asarray(y).ravel()
        return float(np.mean(self.predict(X) == y_true))

    def get_params(self):
        return {
            "kernel": self.kernel,
            "C": self.C,
            "degree": self.degree,
            "coef0": self.coef0,
            "gamma": self.gamma,
            "tol": self.tol,
            "max_passes": self.max_passes,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "cache_size": self.cache_size,
        }

    def save(self, path):
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)
        return path

    @classmethod
    def load(cls, path):
        model = cls()
        with open(Path(path), "rb") as f:
            model.__dict__.update(pickle.load(f))
        return model
