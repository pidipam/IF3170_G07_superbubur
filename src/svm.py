import numpy as np


class SVM:

    def __init__(self, kernel="linear", C=1.0, tol=1e-3, max_passes=5, random_state=42):
        self.kernel = kernel
        self.C = C
        self.tol = tol              
        self.max_passes = max_passes 
        self.random_state = random_state

        self.w = None
        self.b = 0.0
        self.alphas = None
        self.sv_x = None
        self.sv_y = None

    def _compute_kernel(self, X1, X2):
        if self.kernel == "linear":
            return np.dot(X1, X2.T)
        elif self.kernel == "poly":
            return (self._gamma_val * np.dot(X1, X2.T) + self.coef0) ** self.degree
        elif self.kernel == "gauss":
            X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            sq_dists = X1_sq + X2_sq - 2.0 * np.dot(X1, X2.T)
            sq_dists = np.maximum(sq_dists, 0.0)
            return np.exp(-self._gamma_val * sq_dists)
        else:
            raise Exception("kernel name is not valid.")

    def fit(self, X, y):
        if self.gamma == "scale":
            var = X.var()
            self._gamma_val = 1.0 / (n_features * var) if var > 0 else 1.0
        elif self.gamma == "auto":
            self._gamma_val = 1.0 / n_features
        else:
            self._gamma_val = float(self.gamma)
            
        # prediction class should be either 1 or -1 
        # this step could be done in feature eng
        X = np.asarray(X, dtype=float)
        y = np.where(np.asarray(y) <= 0, -1.0, 1.0)

        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        alphas = np.zeros(n_samples, dtype=float)
        b = 0.0
        K = self._compute_kernel(X, X) 

        # SMO
        passes = 0
        while passes < self.max_passes:
            num_changed_alphas = 0

            for i in range(n_samples):
                f_i = np.dot(alphas * y, K[:, i]) + b
                E_i = f_i - y[i]
                if (y[i] * E_i < -self.tol and alphas[i] < self.C) or (
                    y[i] * E_i > self.tol and alphas[i] > 0
                ):
                    j = rng.integers(0, n_samples - 1)
                    if j >= i:
                        j += 1
                    f_j = np.dot(alphas * y, K[:, j]) + b
                    E_j = f_j - y[j]
                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]

                    if y[i] != y[j]:
                        L = max(0.0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0.0, alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])

                    if L == H:
                        continue

                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    alphas[j] = alpha_j_old - (y[j] * (E_i - E_j)) / eta
                    alphas[j] = np.clip(alphas[j], L, H)
                    if abs(alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    alphas[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alphas[j])
                    b1 = (
                        b - E_i - y[i] * (alphas[i] - alpha_i_old) * K[i, i] - y[j] * (alphas[j] - alpha_j_old) * K[i, j]
                    )
                    b2 = (
                        b - E_j - y[i] * (alphas[i] - alpha_i_old) * K[i, j] - y[j] * (alphas[j] - alpha_j_old) * K[j, j]
                    )

                    if 0 < alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0

                    num_changed_alphas += 1
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        sv_mask = alphas > 1e-5
        self.alphas = alphas[sv_mask]
        self.sv_x = X[sv_mask]
        self.sv_y = y[sv_mask]
        self.b = b

        # for linear kernel, w can be computed directly
        if self.kernel == "linear":
            self.w = np.dot(self.alphas * self.sv_y, self.sv_x)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        K_eval = self._compute_kernel(X, self.sv_x)
        scores = np.dot(K_eval, self.alphas * self.sv_y) + self.b
        return np.where(scores >= 0, 1, -1)