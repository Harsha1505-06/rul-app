"""Data loading, labels, features and metrics for C-MAPSS."""
import numpy as np, pandas as pd

COLS = ["engine_id", "cycle"] + [f"setting{i}" for i in (1, 2, 3)] + [f"s{i}" for i in range(1, 22)]
DROP = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]  # near-constant in FD001
CLIP, WINDOWS = 125, (10, 30)
STATES = ["Healthy", "Early wear", "Advanced degradation", "Near-failure"]


def load(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS).sort_values(["engine_id", "cycle"]).reset_index(drop=True)


def sensor_cols():
    return [c for c in COLS[5:] if c not in DROP]


def add_rul(df, clip=CLIP):
    mx = df.groupby("engine_id")["cycle"].transform("max")
    return df.assign(rul=(mx - df["cycle"]).clip(upper=clip))


def rolling_features(df):
    """Trailing windows only (min_periods=1): row t never sees cycles > t."""
    s, g = sensor_cols(), df.groupby("engine_id")[sensor_cols()]
    parts = [df[["engine_id", "cycle"]], df[s]]
    for w in WINDOWS:
        r = g.rolling(w, min_periods=1)
        parts.append(r.mean().droplevel(0).sort_index().add_suffix(f"_m{w}"))
        parts.append(r.std().droplevel(0).sort_index().fillna(0).add_suffix(f"_sd{w}"))
    return pd.concat(parts, axis=1)


def nasa_score(y_true, y_pred):
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)  # d = predicted - true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))
