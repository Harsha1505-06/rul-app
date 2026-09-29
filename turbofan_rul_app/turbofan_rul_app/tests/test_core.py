import numpy as np, pandas as pd
from src.core import add_rul, nasa_score, rolling_features, COLS


def fake(n=2, L=40):
    rng = np.random.default_rng(0)
    return pd.DataFrame([[e, t] + list(rng.normal(size=24)) for e in range(1, n + 1) for t in range(1, L + 1)], columns=COLS)


def test_rul_clipped():
    r = add_rul(fake(1, 200))
    assert r.rul.iloc[0] == 125 and r.rul.iloc[-1] == 0


def test_nasa_score_hand_values():
    assert np.isclose(nasa_score([100], [90]), np.exp(10 / 13) - 1)   # early
    assert np.isclose(nasa_score([100], [110]), np.exp(1) - 1)        # late


def test_rolling_no_future_leakage():
    a = fake(); b = a.copy(); b.loc[b.cycle > 20, "s2"] += 100
    fa, fb = rolling_features(a), rolling_features(b)
    cols = [c for c in fa.columns if c.startswith("s2")]
    assert np.allclose(fa.loc[a.cycle <= 20, cols], fb.loc[b.cycle <= 20, cols])
