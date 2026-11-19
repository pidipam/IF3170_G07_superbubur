import json
from pathlib import Path
import numpy as np
import pandas as pd

TARGET = "is_canceled"
ID_COL = "id"

CATEGORICAL_COLS = ["meal_package", "room_category", "booking_channel"]
# Kolom numerik mentah (termasuk biner 0/1 yang berisi null).
NUMERIC_COLS = [
    "num_adults",
    "num_children",
    "weekend_nights",
    "weekday_nights",
    "needs_parking",
    "days_before_arrival",
    "arrival_year",
    "arrival_month",
    "arrival_day",
    "is_returning_guest",
    "prior_cancellations",
    "prior_completed_bookings",
    "nightly_rate_idr",
    "special_requests",
]
UNKNOWN = "Unknown"

# Konfigurasi final preprocessing
DEFAULT_PREP_PARAMS = dict(
    scaling="standard",
    encoding="onehot",
    categorical_missing="mode",
    clip_quantiles=(0.01, 0.99),
    log_cols=("prior_cancellations", "prior_completed_bookings", "nightly_rate_idr"),
    drop_features=("arrival_month",),  # sudah diwakili arrival_month_sin/cos
)


# Cleaning & splitting
def remove_duplicates(df: pd.DataFrame, id_col: str = ID_COL) -> pd.DataFrame:
    subset = [c for c in df.columns if c != id_col]
    return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)


def stratified_split(df: pd.DataFrame, target: str = TARGET, val_size: float = 0.2, seed: int = 42):
    if not 0 < val_size < 1:
        raise ValueError("val_size harus di antara 0 dan 1")
    rng = np.random.default_rng(seed)
    val_idx = []
    for _, group in df.groupby(target):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        val_idx.extend(idx[: int(round(len(idx) * val_size))])
    mask = df.index.isin(val_idx)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


# Preprocessor
class Preprocessor:
    CLIP_COLS = [
        "num_children",
        "weekend_nights",
        "weekday_nights",
        "total_nights",
        "total_guests",
        "prior_cancellations",
        "prior_completed_bookings",
        "nightly_rate_idr",
        "days_before_arrival",
    ]
    BINARY_COLS = ["needs_parking", "is_returning_guest"]

    def __init__(
        self,
        scaling: str | None = "standard",
        encoding: str = "onehot",
        categorical_missing: str = "mode",
        clip_quantiles: tuple | None = (0.01, 0.99),
        log_cols: tuple = ("prior_cancellations", "prior_completed_bookings", "nightly_rate_idr"),
        drop_features: tuple = (),
    ):
        if scaling not in ("minmax", "standard", None):
            raise ValueError("scaling harus 'minmax', 'standard', atau None")
        if encoding not in ("onehot", "ordinal"):
            raise ValueError("encoding harus 'onehot' atau 'ordinal'")
        if categorical_missing not in ("unknown", "mode"):
            raise ValueError("categorical_missing harus 'unknown' atau 'mode'")
        self.scaling = scaling
        self.encoding = encoding
        self.categorical_missing = categorical_missing
        self.clip_quantiles = None if clip_quantiles is None else tuple(clip_quantiles)
        self.log_cols = tuple(log_cols)
        self.drop_features = tuple(drop_features)

        self.state_: dict | None = None
        self.feature_names_: list[str] | None = None
        self.categorical_idx_: list[int] = []

    # Helper internal
    @staticmethod
    def _cast(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in NUMERIC_COLS:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        for c in CATEGORICAL_COLS:
            out[c] = out[c].astype("object").where(out[c].notna(), np.nan)
            out[c] = out[c].map(lambda v: v.strip() if isinstance(v, str) else v)
        return out

    @staticmethod
    def _construct(df: pd.DataFrame) -> pd.DataFrame:
        """Atribut turunan. Dijalankan setelah imputasi sehingga tidak ada NaN."""
        out = df.copy()
        out["total_nights"] = out["weekend_nights"] + out["weekday_nights"]
        out["total_guests"] = out["num_adults"] + out["num_children"]
        date = pd.to_datetime(
            dict(year=out["arrival_year"], month=out["arrival_month"], day=out["arrival_day"]),
            errors="coerce",
        )
        out["arrival_dow"] = date.dt.dayofweek.fillna(0).astype(float)
        angle = 2.0 * np.pi * (out["arrival_month"] - 1) / 12.0
        out["arrival_month_sin"] = np.sin(angle)
        out["arrival_month_cos"] = np.cos(angle)
        return out

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        s = self.state_
        out = df.copy()
        for c in NUMERIC_COLS:
            if c == "nightly_rate_idr":
                by_channel = out["booking_channel"].map(s["rate_median_by_channel"])
                out[c] = out[c].fillna(by_channel).fillna(s["median"][c])
            else:
                out[c] = out[c].fillna(s["median"][c])
        for c in CATEGORICAL_COLS:
            fill = UNKNOWN if self.categorical_missing == "unknown" else s["mode"][c]
            out[c] = out[c].fillna(fill)
            # kategori yang tidak pernah dilihat saat fit -> "Unknown"
            out[c] = out[c].where(out[c].isin(s["categories"][c]), UNKNOWN)
        return out

    def _numeric_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        s = self.state_
        out = df[s["numeric_features"]].astype(float).copy()
        if self.clip_quantiles is not None:
            for c, (lo, hi) in s["clip_bounds"].items():
                out[c] = out[c].clip(lo, hi)
        for c in self.log_cols:
            out[c] = np.log1p(out[c].clip(lower=0))
        if self.scaling is not None:
            out = (out - pd.Series(s["scale_shift"])) / pd.Series(s["scale_div"])
        return out

    # API
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """Hitung seluruh statistik dari data latih ``df`` (boleh memuat kolom target/id)."""
        df = self._cast(df)
        state: dict = {"median": {}, "mode": {}, "categories": {}}
        for c in NUMERIC_COLS:
            state["median"][c] = float(df[c].median())
        state["rate_median_by_channel"] = {
            k: float(v) for k, v in df.groupby("booking_channel")["nightly_rate_idr"].median().dropna().items()
        }
        for c in CATEGORICAL_COLS:
            vc = df[c].dropna()
            state["mode"][c] = str(vc.mode().iloc[0])
            state["categories"][c] = sorted(map(str, vc.unique())) + [UNKNOWN]
        self.state_ = state

        filled = self._construct(self._impute(df))
        numeric_features = NUMERIC_COLS + [
            "total_nights",
            "total_guests",
            "arrival_dow",
            "arrival_month_sin",
            "arrival_month_cos",
        ]
        state["numeric_features"] = numeric_features

        state["clip_bounds"] = {}
        if self.clip_quantiles is not None:
            lo_q, hi_q = self.clip_quantiles
            for c in self.CLIP_COLS:
                lo, hi = float(filled[c].quantile(lo_q)), float(filled[c].quantile(hi_q))
                if hi > lo:  # jika batas menyusut jadi satu nilai, clipping akan menghapus fitur 
                    state["clip_bounds"][c] = (lo, hi)

        # statistik scaling dihitung setelah clipping dan log
        tmp = filled[numeric_features].astype(float).copy()
        for c, (lo, hi) in state["clip_bounds"].items():
            tmp[c] = tmp[c].clip(lo, hi)
        for c in self.log_cols:
            tmp[c] = np.log1p(tmp[c].clip(lower=0))
        if self.scaling == "minmax":
            shift, div = tmp.min(), (tmp.max() - tmp.min())
        elif self.scaling == "standard":
            shift, div = tmp.mean(), tmp.std(ddof=0)
        else:
            shift, div = pd.Series(0.0, index=tmp.columns), pd.Series(1.0, index=tmp.columns)
        div = div.replace(0, 1.0)  # kolom konstan -> 0 setelah scaling
        state["scale_shift"] = {k: float(v) for k, v in shift.items()}
        state["scale_div"] = {k: float(v) for k, v in div.items()}

        self._build_feature_names()
        return self

    def _build_feature_names(self) -> None:
        s = self.state_
        names = list(s["numeric_features"])
        if self.encoding == "onehot":
            for c in CATEGORICAL_COLS:
                names += [f"{c}={cat}" for cat in s["categories"][c]]
        else:
            names += list(CATEGORICAL_COLS)
        keep = [i for i, n in enumerate(names) if n not in self.drop_features]
        s["keep_idx"] = keep
        self.feature_names_ = [names[i] for i in keep]
        if self.encoding == "ordinal":
            self.categorical_idx_ = [i for i, n in enumerate(self.feature_names_) if n in CATEGORICAL_COLS]
        else:
            self.categorical_idx_ = []

    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Seperti ``transform`` tetapi mengembalikan DataFrame dengan nama kolom."""
        if self.state_ is None:
            raise RuntimeError("Preprocessor belum di-fit")
        s = self.state_
        filled = self._construct(self._impute(self._cast(df)))
        num = self._numeric_matrix(filled)
        parts = [num]
        for c in CATEGORICAL_COLS:
            cats = s["categories"][c]
            if self.encoding == "onehot":
                parts.append(
                    pd.DataFrame(
                        {f"{c}={cat}": (filled[c] == cat).astype(float) for cat in cats}, index=filled.index
                    )
                )
            else:
                parts.append(pd.DataFrame({c: filled[c].map({k: float(i) for i, k in enumerate(cats)})}))
        full = pd.concat(parts, axis=1)
        return full.iloc[:, s["keep_idx"]].reset_index(drop=True)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Matriks fitur float64 siap pakai (tanpa NaN)."""
        return self.transform_frame(df).to_numpy(dtype=float)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # Persistence
    def get_params(self) -> dict:
        return {
            "scaling": self.scaling,
            "encoding": self.encoding,
            "categorical_missing": self.categorical_missing,
            "clip_quantiles": self.clip_quantiles,
            "log_cols": list(self.log_cols),
            "drop_features": list(self.drop_features),
        }

    def save(self, path) -> Path:
        """Simpan parameter + statistik hasil fit ke JSON (mudah diperiksa manusia)."""
        if self.state_ is None:
            raise RuntimeError("Preprocessor belum di-fit")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"params": self.get_params(), "state": self.state_}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "Preprocessor":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        params = payload["params"]
        params["clip_quantiles"] = None if params["clip_quantiles"] is None else tuple(params["clip_quantiles"])
        params["log_cols"] = tuple(params["log_cols"])
        params["drop_features"] = tuple(params["drop_features"])
        prep = cls(**params)
        state = payload["state"]
        state["clip_bounds"] = {k: tuple(v) for k, v in state["clip_bounds"].items()}
        prep.state_ = state
        prep._build_feature_names()
        return prep


# Penanganan imbalance (hanya untuk data latih!)
def random_oversample(X, y, seed: int = 42):
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()
    idx = [np.arange(len(y))]
    for c, n in zip(classes, counts):
        if n < target:
            idx.append(rng.choice(np.where(y == c)[0], size=target - n, replace=True))
    idx = np.concatenate(idx)
    return X[idx], y[idx]


def random_undersample(X, y, seed: int = 42):   
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.min()
    idx = np.concatenate([rng.choice(np.where(y == c)[0], size=target, replace=False) for c in classes])
    idx.sort()
    return X[idx], y[idx]


def smote(X, y, k: int = 5, seed: int = 42):
    from src.knn import KNN  # import lokal 

    X, y = np.asarray(X, dtype=float), np.asarray(y)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()
    new_X, new_y = [X], [y]
    for c, n in zip(classes, counts):
        n_new = target - n
        if n_new <= 0:
            continue
        Xc = X[y == c]
        if len(Xc) < 2:
            raise ValueError("SMOTE membutuhkan minimal 2 sampel pada kelas minoritas")
        kk = min(k, len(Xc) - 1)
        nn = KNN(k=kk + 1).fit(Xc, np.zeros(len(Xc), dtype=int))
        _, idx = nn.kneighbors(Xc, k=kk + 1)
        # buang diri sendiri dari daftar tetangga berdasarkan indeks
        neigh = np.empty((len(Xc), kk), dtype=int)
        for i in range(len(Xc)):
            row = idx[i][idx[i] != i]
            neigh[i] = row[:kk]
        base = rng.integers(0, len(Xc), size=n_new)
        pick = neigh[base, rng.integers(0, kk, size=n_new)]
        u = rng.random((n_new, 1))
        new_X.append(Xc[base] + u * (Xc[pick] - Xc[base]))
        new_y.append(np.full(n_new, c, dtype=y.dtype))
    return np.vstack(new_X), np.concatenate(new_y)

# Final
def prepare_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame, val_size: float = 0.2, seed: int = 42,
                     prep_params: dict | None = None) -> dict:
    params = dict(DEFAULT_PREP_PARAMS if prep_params is None else prep_params)
    train_set, val_set = stratified_split(train_df, target=TARGET, val_size=val_size, seed=seed)
    train_set = remove_duplicates(train_set)
    prep = Preprocessor(**params).fit(train_set)
    return {
        "prep": prep,
        "train_set": train_set,
        "val_set": val_set,
        "X_train": prep.transform(train_set),
        "y_train": train_set[TARGET].to_numpy(),
        "X_val": prep.transform(val_set),
        "y_val": val_set[TARGET].to_numpy(),
        "X_test": prep.transform(test_df),
    }