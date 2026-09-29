"""End-to-end pipeline: python -m src.train [--raw data/raw] [--out artifacts]"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from .core import CLIP, STATES, add_rul, load, nasa_score, rmse, rolling_features


def onehot(lab, k=4):
    return np.eye(k)[lab]


def main(raw="data/raw", out="artifacts", seed=42, k=4, boot=2000):
    raw, out = Path(raw), Path(out); out.mkdir(parents=True, exist_ok=True)
    tr = add_rul(load(raw / "train_FD001.txt"))
    Ftr = rolling_features(tr)
    Fte = rolling_features(load(raw / "test_FD001.txt")).groupby("engine_id").tail(1)  # last cycle per engine
    y_te = np.minimum(np.loadtxt(raw / "RUL_FD001.txt"), CLIP)
    feats = [c for c in Ftr.columns if c not in ("engine_id", "cycle")]
    sc = StandardScaler().fit(Ftr[feats])                      # fit on train only
    Ztr, Zte = sc.transform(Ftr[feats]), sc.transform(Fte[feats])
    y = tr["rul"].values

    # unsupervised states: no RUL used to fit; RUL only orders clusters (train labels)
    km = KMeans(k, n_init=10, random_state=seed).fit(Ztr)
    order = np.argsort(-pd.Series(y).groupby(km.labels_).mean().values)  # healthiest first
    remap = np.empty(k, int); remap[order] = np.arange(k)
    ltr, lte = remap[km.labels_], remap[km.predict(Zte)]
    rng = np.random.default_rng(seed)
    rand_tr, rand_te = rng.permutation(ltr), rng.choice(ltr, size=len(lte))  # control

    def fit_pred(Xa, Xb):
        m = RandomForestRegressor(200, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=seed).fit(Xa, y)
        return np.clip(m.predict(Xb), 0, CLIP)

    preds = {"Baseline (raw + rolling)": fit_pred(Ztr, Zte),
             "+ Discovered states": fit_pred(np.hstack([Ztr, onehot(ltr, k)]), np.hstack([Zte, onehot(lte, k)])),
             "Control: random labels": fit_pred(np.hstack([Ztr, onehot(rand_tr, k)]), np.hstack([Zte, onehot(rand_te, k)]))}
    models = {n: {"rmse": rmse(y_te, p), "nasa_score": nasa_score(y_te, p)} for n, p in preds.items()}

    # bootstrap over test engines: RMSE(baseline) - RMSE(model); >0 means model better
    names, base = list(preds), preds["Baseline (raw + rolling)"]
    ci = {}
    for n in names[1:]:
        idx = rng.integers(0, len(y_te), (boot, len(y_te)))
        d = [rmse(y_te[i], base[i]) - rmse(y_te[i], preds[n][i]) for i in idx]
        ci[n] = {"delta_rmse": models[names[0]]["rmse"] - models[n]["rmse"], "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}

    pca = PCA(3, random_state=seed).fit(Ztr); P = pca.transform(Ztr)
    lat = pd.DataFrame({"engine_id": tr.engine_id, "cycle": tr.cycle, "rul": y, "state": [STATES[i] for i in ltr],
                        "pc1": P[:, 0], "pc2": P[:, 1], "pc3": P[:, 2]})
    lat.to_parquet(out / "latent_data.parquet")
    pd.DataFrame({"engine_id": Fte.engine_id.values, "true_rul": y_te, **{n: p for n, p in preds.items()}}).to_parquet(out / "test_predictions.parquet")
    res = {"dataset": "FD001", "k": k, "seed": seed, "n_test_engines": int(len(y_te)), "models": models, "delta_vs_baseline": ci,
           "state_profile": {STATES[i]: float(y[ltr == i].mean()) for i in range(k)},
           "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_]}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--raw", default="data/raw"); a.add_argument("--out", default="artifacts")
    print(json.dumps(main(**vars(a.parse_args())), indent=2))
