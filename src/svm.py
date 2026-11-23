import pickle
from pathlib import Path
import numpy as np
from scipy.optimize import Bounds, minimize


class SVM:

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
        self._plot_projector_ = None

    @property
    def _is_gauss(self) -> bool:
        return self.kernel in ("gauss", "rbf")

    @staticmethod
    def _fast_power_inplace(arr, degree):
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

    def _build_2d_projector(self, X, feature_idx=None, feature_names=None):
        n_features = X.shape[1]
        if n_features < 2:
            raise ValueError(
                f"Visualisasi 2D membutuhkan minimal 2 fitur (ditemukan {n_features})."
            )

        def _feat_label(idx):
            if feature_names is not None:
                return str(feature_names[idx])
            return f"x{idx + 1}"

        if n_features == 2 and feature_idx is None:
            return {
                "mode": "direct",
                "x_label": _feat_label(0),
                "y_label": _feat_label(1),
            }

        if feature_idx is not None:
            if len(feature_idx) != 2:
                raise ValueError("feature_idx harus berisi tepat 2 indeks fitur, mis. (0, 1).")
            i, j = int(feature_idx[0]), int(feature_idx[1])
            if not (0 <= i < n_features and 0 <= j < n_features) or i == j:
                raise ValueError(
                    f"feature_idx={(i, j)} tidak valid untuk {n_features} fitur."
                )
            return {
                "mode": "slice",
                "feature_idx": (i, j),
                "base_point": np.median(X, axis=0),
                "x_label": _feat_label(i),
                "y_label": _feat_label(j),
            }

        mean = np.mean(X, axis=0)
        X_centered = X - mean
        _, s, vt = np.linalg.svd(X_centered, full_matrices=False)
        components = vt[:2]
        var = (s ** 2) / max(len(X) - 1, 1)
        total_var = float(np.sum(var))
        ratios = (var[:2] / total_var) if total_var > 0.0 else np.zeros(2)
        return {
            "mode": "pca",
            "mean": mean,
            "components": components,
            "x_label": f"PC1 ({ratios[0] * 100:.1f}%)",
            "y_label": f"PC2 ({ratios[1] * 100:.1f}%)",
        }

    @staticmethod
    def _project_to_2d(X, projector):
        mode = projector["mode"]
        if mode == "direct":
            return X[:, :2]
        if mode == "slice":
            i, j = projector["feature_idx"]
            return X[:, [i, j]]
        return (X - projector["mean"]) @ projector["components"].T

    @staticmethod
    def _lift_from_2d(Z_2d, projector):
        mode = projector["mode"]
        if mode == "direct":
            return Z_2d
        if mode == "slice":
            i, j = projector["feature_idx"]
            X_lifted = np.tile(projector["base_point"], (Z_2d.shape[0], 1))
            X_lifted[:, i] = Z_2d[:, 0]
            X_lifted[:, j] = Z_2d[:, 1]
            return X_lifted
        return Z_2d @ projector["components"] + projector["mean"]

    def plot_decision_boundary(
        self,
        X_train,
        y_train,
        ax=None,
        feature_idx=None,
        feature_names=None,
        class_names=None,
        grid_resolution=200,
        padding=0.5,
        show_margins=True,
        show_support_vectors=True,
        max_train_points=None,
        title=None,
        save_path=None,
        figsize=(8, 6),
    ):
        import matplotlib.pyplot as plt

        self._check_fitted()
        X_tr = np.asarray(X_train, dtype=np.float64)
        y_tr = np.asarray(y_train).ravel()
        if X_tr.ndim != 2 or X_tr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X_train harus 2D dengan {self.n_features_in_} fitur, didapat shape {X_tr.shape}."
            )
        if X_tr.shape[0] != y_tr.shape[0]:
            raise ValueError("Jumlah baris X_train dan y_train tidak sama.")

        projector = self._build_2d_projector(
            X_tr, feature_idx=feature_idx, feature_names=feature_names
        )
        self._plot_projector_ = projector

        Z_tr = self._project_to_2d(X_tr, projector)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        ax._svm_projector_ = projector

        x_min, x_max = float(Z_tr[:, 0].min()), float(Z_tr[:, 0].max())
        y_min, y_max = float(Z_tr[:, 1].min()), float(Z_tr[:, 1].max())
        dx = max(x_max - x_min, 1.0)
        dy = max(y_max - y_min, 1.0)
        pad_x = padding if isinstance(padding, (int, float)) else padding[0]
        pad_y = padding if isinstance(padding, (int, float)) else padding[1]
        pad_x = max(float(pad_x), 0.08 * dx)
        pad_y = max(float(pad_y), 0.08 * dy)

        xx, yy = np.meshgrid(
            np.linspace(x_min - pad_x, x_max + pad_x, int(grid_resolution)),
            np.linspace(y_min - pad_y, y_max + pad_y, int(grid_resolution)),
        )
        grid_2d = np.column_stack([xx.ravel(), yy.ravel()])
        grid_full = self._lift_from_2d(grid_2d, projector)
        Z_scores = self.decision_function(grid_full).reshape(xx.shape)
        z_min, z_max = float(Z_scores.min()), float(Z_scores.max())
        if z_min < 0.0 < z_max:
            ax.contourf(
                xx,
                yy,
                Z_scores,
                levels=[z_min - 1.0, 0.0, z_max + 1.0],
                colors=["#dce8f7", "#fbdcdc"],
                alpha=0.55,
                zorder=0,
            )
            ax.contour(
                xx,
                yy,
                Z_scores,
                levels=[0.0],
                colors=["#111111"],
                linewidths=2.0,
                linestyles="-",
                zorder=3,
            )
            ax.plot([], [], color="#111111", lw=2.0, ls="-", label="Decision Boundary (f(x)=0)")

        if show_margins:
            margin_levels = [lv for lv in (-1.0, 1.0) if z_min < lv < z_max]
            if margin_levels:
                ax.contour(
                    xx,
                    yy,
                    Z_scores,
                    levels=margin_levels,
                    colors=["#555555"],
                    linewidths=1.2,
                    linestyles="--",
                    zorder=3,
                )
                ax.plot([], [], color="#555555", lw=1.2, ls="--", label="Margin (f(x)=±1)")

        if max_train_points is not None and len(Z_tr) > int(max_train_points):
            rng = np.random.default_rng(self.random_state)
            idx_plot = rng.choice(len(Z_tr), size=int(max_train_points), replace=False)
            Z_plot, y_plot = Z_tr[idx_plot], y_tr[idx_plot]
        else:
            Z_plot, y_plot = Z_tr, y_tr

        c0, c1 = self.classes_[0], self.classes_[1]
        lbl0 = f"Train Kelas {class_names[0]}" if class_names is not None else f"Train Kelas {c0}"
        lbl1 = f"Train Kelas {class_names[1]}" if class_names is not None else f"Train Kelas {c1}"

        mask0 = y_plot == c0
        mask1 = y_plot == c1
        ax.scatter(
            Z_plot[mask0, 0],
            Z_plot[mask0, 1],
            c="#1f77b4",
            marker="o",
            s=36,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.4,
            label=lbl0,
            zorder=4,
        )
        ax.scatter(
            Z_plot[mask1, 0],
            Z_plot[mask1, 1],
            c="#d62728",
            marker="^",
            s=42,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.4,
            label=lbl1,
            zorder=4,
        )

        if show_support_vectors and len(self.sv_x) > 0:
            Z_sv = self._project_to_2d(self.sv_x, projector)
            ax.scatter(
                Z_sv[:, 0],
                Z_sv[:, 1],
                s=110,
                facecolors="none",
                edgecolors="#f59e0b",
                linewidths=1.4,
                label=f"Support Vectors ({len(self.sv_x)})",
                zorder=5,
            )

        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_xlabel(projector["x_label"])
        ax.set_ylabel(projector["y_label"])
        if title is None:
            mode_note = ""
            if projector["mode"] == "pca":
                mode_note = " — Proyeksi PCA 2D"
            elif projector["mode"] == "slice":
                mode_note = " — Irisan 2 Fitur"
            title = f"SVM Decision Boundary (kernel='{self.kernel}', C={self.C:g}){mode_note}"
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="best", fontsize=8.5, framealpha=0.9)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig, ax

    def plot_test_points(
        self,
        ax,
        X_test,
        y_test=None,
        annotate="auto",
        show_scores=False,
        marker="*",
        size=180,
        save_path=None,
    ):
        self._check_fitted()
        X_te = np.asarray(X_test, dtype=np.float64)
        if X_te.ndim == 1:
            X_te = X_te.reshape(1, -1)
        if X_te.ndim != 2 or X_te.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X_test harus memiliki {self.n_features_in_} fitur, didapat shape {X_te.shape}."
            )

        projector = getattr(ax, "_svm_projector_", None) or self._plot_projector_
        if projector is None:
            if self.n_features_in_ == 2:
                projector = {"mode": "direct", "x_label": "x1", "y_label": "x2"}
            else:
                raise RuntimeError(
                    "Proyektor 2D belum tersedia. Panggil plot_decision_boundary(...) "
                    "terlebih dahulu sebelum plot_test_points(...)."
                )

        Z_te = self._project_to_2d(X_te, projector)
        y_pred = self.predict(X_te)
        scores = self.decision_function(X_te)

        c0, c1 = self.classes_[0], self.classes_[1]

        if y_test is None:
            m0 = y_pred == c0
            m1 = y_pred == c1
            if np.any(m0):
                ax.scatter(
                    Z_te[m0, 0],
                    Z_te[m0, 1],
                    c="#00b4d8",
                    marker=marker,
                    s=size,
                    edgecolors="#111111",
                    linewidths=1.0,
                    label=f"Uji (Pred {c0})",
                    zorder=7,
                )
            if np.any(m1):
                ax.scatter(
                    Z_te[m1, 0],
                    Z_te[m1, 1],
                    c="#ff4d6d",
                    marker=marker,
                    s=size,
                    edgecolors="#111111",
                    linewidths=1.0,
                    label=f"Uji (Pred {c1})",
                    zorder=7,
                )
        else:
            y_true = np.asarray(y_test).ravel()
            if len(y_true) != len(X_te):
                raise ValueError("Jumlah elemen y_test dan baris X_test tidak sama.")
            correct = y_pred == y_true
            wrong = ~correct

            if np.any(correct):
                ax.scatter(
                    Z_te[correct, 0],
                    Z_te[correct, 1],
                    c="#10b981",
                    marker=marker,
                    s=size,
                    edgecolors="#111111",
                    linewidths=1.0,
                    label=f"Uji Benar ({int(np.sum(correct))})",
                    zorder=7,
                )
            if np.any(wrong):
                ax.scatter(
                    Z_te[wrong, 0],
                    Z_te[wrong, 1],
                    c="#facc15",
                    marker="X",
                    s=size * 0.85,
                    edgecolors="#b91c1c",
                    linewidths=1.4,
                    label=f"Uji Salah ({int(np.sum(wrong))})",
                    zorder=8,
                )

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        x_span = max(cur_xlim[1] - cur_xlim[0], 1e-9)
        y_span = max(cur_ylim[1] - cur_ylim[0], 1e-9)
        te_xmin, te_xmax = float(Z_te[:, 0].min()), float(Z_te[:, 0].max())
        te_ymin, te_ymax = float(Z_te[:, 1].min()), float(Z_te[:, 1].max())

        new_xmin = min(cur_xlim[0], te_xmin - 0.08 * x_span) if te_xmin < cur_xlim[0] else cur_xlim[0]
        new_xmax = max(cur_xlim[1], te_xmax + 0.08 * x_span) if te_xmax > cur_xlim[1] else cur_xlim[1]
        new_ymin = min(cur_ylim[0], te_ymin - 0.08 * y_span) if te_ymin < cur_ylim[0] else cur_ylim[0]
        new_ymax = max(cur_ylim[1], te_ymax + 0.08 * y_span) if te_ymax > cur_ylim[1] else cur_ylim[1]
        ax.set_xlim(new_xmin, new_xmax)
        ax.set_ylim(new_ymin, new_ymax)

        do_annotate = (len(Z_te) <= 10) if annotate == "auto" else bool(annotate)
        if do_annotate:
            y_true_arr = None if y_test is None else np.asarray(y_test).ravel()
            x_right_thresh = new_xmin + 0.68 * (new_xmax - new_xmin)
            y_top_thresh = new_ymin + 0.85 * (new_ymax - new_ymin)
            for idx in range(len(Z_te)):
                parts = [f"T{idx + 1}: pred={y_pred[idx]}"]
                if y_true_arr is not None:
                    parts.append(f"true={y_true_arr[idx]}")
                if show_scores:
                    parts.append(f"f={scores[idx]:.2f}")
                px, py = float(Z_te[idx, 0]), float(Z_te[idx, 1])
                ha = "right" if px > x_right_thresh else "left"
                dx_pt = -7 if px > x_right_thresh else 7
                dy_pt = -14 if py > y_top_thresh else 7
                ax.annotate(
                    " | ".join(parts),
                    (px, py),
                    textcoords="offset points",
                    xytext=(dx_pt, dy_pt),
                    ha=ha,
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#444444", alpha=0.85),
                    zorder=9,
                )

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="best", fontsize=8.5, framealpha=0.9)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")

        return ax

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
