import pickle
import numpy as np

class LogisticRegression:

    def __init__(self, penalty="l2", lam=0.01, l1_ratio=0.5, lr=0.01, n_iter=50, batch_size=1, tol=1e-6, fit_intercept=True, class_weight=None, random_state=42, shuffle=True, record_history=False):
        self.penalty = penalty
        self.lam = lam
        self.l1_ratio = l1_ratio
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.class_weight = class_weight
        self.random_state = random_state
        self.shuffle = shuffle
        self.record_history = record_history
        self.coef_ = None
        self.intercept_ = 0.0
        self.n_iter_ = 0
        self.loss_history = []
        self.theta_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _reg_strength(self):
        if self.penalty is None or self.penalty == "none":
            return 0.0, 0.0
        if self.penalty == "l1":
            return self.lam, 0.0
        if self.penalty == "l2":
            return 0.0, self.lam
        if self.penalty == "elasticnet":
            return self.lam * self.l1_ratio, self.lam * (1.0 - self.l1_ratio)
        raise ValueError("penalty must be None, 'l1', 'l2', or 'elasticnet'.")

    def _sample_weight(self, y):
        if self.class_weight is None:
            return np.ones(len(y))
        if self.class_weight == "balanced":
            n = len(y)
            n1 = max(np.sum(y == 1), 1)
            n0 = max(np.sum(y == 0), 1)
            return np.where(y == 1, n / (2.0 * n1), n / (2.0 * n0))
        return np.where(y == 1, self.class_weight.get(1, 1.0), self.class_weight.get(0, 1.0))

    def _loss(self, X, y, sw, w, b):
        p = np.clip(self._sigmoid(X @ w + b), 1e-12, 1 - 1e-12)
        a1, a2 = self._reg_strength()
        ll = -np.sum(sw * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(sw)
        return ll + a1 * np.sum(np.abs(w)) + 0.5 * a2 * np.sum(w ** 2)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = (np.asarray(y).ravel() > 0).astype(float)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)
        a1, a2 = self._reg_strength()
        sw = self._sample_weight(y)
        bs = n if self.batch_size is None else min(self.batch_size, n)
        w = np.zeros(d)
        b = 0.0
        self.loss_history = [self._loss(X, y, sw, w, b)]
        self.theta_history = [np.r_[b, w]] if self.record_history else []
        epoch = 0
        for epoch in range(self.n_iter):
            idx = rng.permutation(n) if self.shuffle else np.arange(n)
            for s in range(0, n, bs):
                bi = idx[s:s + bs]
                p = self._sigmoid(X[bi] @ w + b)
                r = sw[bi] * (y[bi] - p)
                w = w + self.lr * (X[bi].T @ r / len(bi) - a2 * w)
                if a1 > 0:
                    w = np.sign(w) * np.maximum(np.abs(w) - self.lr * a1, 0.0)
                if self.fit_intercept:
                    b = b + self.lr * np.sum(r) / len(bi)
            loss = self._loss(X, y, sw, w, b)
            self.loss_history.append(loss)
            if self.record_history:
                self.theta_history.append(np.r_[b, w])
            if abs(self.loss_history[-2] - loss) < self.tol:
                break
        self.coef_ = w
        self.intercept_ = b
        self.n_iter_ = epoch + 1
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def predict_probability(self, X):
        p = self._sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold=0.5):
        return (self.predict_probability(X)[:, 1] >= threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == (np.asarray(y).ravel() > 0).astype(int))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        model = cls()
        with open(path, "rb") as f:
            model.__dict__.update(pickle.load(f))
        return model