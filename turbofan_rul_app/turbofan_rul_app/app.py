import json
from pathlib import Path
import pandas as pd, plotly.express as px, streamlit as st

ART = Path(__file__).parent / "artifacts"
STATES = ["Healthy", "Early wear", "Advanced degradation", "Near-failure"]
COL = dict(zip(STATES, ["#1a9e6a", "#d4a814", "#e07a1f", "#d63b3b"]))
st.set_page_config(page_title="Latent Engine Health States", page_icon="🛩️", layout="wide")
st.title("Latent Engine Health States")
st.caption("Unsupervised health states vs. supervised RUL prediction · NASA C-MAPSS FD001")

if not (ART / "metrics.json").exists():
    st.error("No artifacts found. Place C-MAPSS files in data/raw/ and run `python -m src.train`.")
    st.stop()


@st.cache_data
def load():
    return json.loads((ART / "metrics.json").read_text()), pd.read_parquet(ART / "latent_data.parquet"), pd.read_parquet(ART / "test_predictions.parquet")


m, lat, tp = load()
t1, t2, t3 = st.tabs(["Results", "Latent space", "Engine explorer"])

with t1:
    st.subheader("Official test protocol: last cycle of each test engine")
    df = pd.DataFrame(m["models"]).T.rename(columns={"rmse": "RMSE", "nasa_score": "NASA score"})
    st.dataframe(df.style.format("{:.2f}"), use_container_width=True)
    for name, d in m["delta_vs_baseline"].items():
        sig = d["lo"] > 0
        st.metric(f"{name}: RMSE improvement vs baseline", f"{d['delta_rmse']:+.2f}", f"95% CI [{d['lo']:.2f}, {d['hi']:.2f}] over {m['n_test_engines']} engines")
        st.write("Interval excludes zero." if sig else "Interval includes zero: no reliable difference.")
    st.caption("Gains only count if they exceed the random-label control. Positive = better than baseline (lower RMSE).")
    st.plotly_chart(px.bar(pd.Series(m["state_profile"], name="mean train RUL").reset_index(), x="index", y="mean train RUL", color="index", color_discrete_map=COL, labels={"index": "State"}), use_container_width=True)
    fig = px.scatter(tp, x="true_rul", y="+ Discovered states", labels={"true_rul": "True RUL", "+ Discovered states": "Predicted RUL"})
    fig.add_shape(type="line", x0=0, y0=0, x1=125, y1=125, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

with t2:
    st.caption(f"PCA of training features, coloured by discovered state (explained variance {sum(m['pca_explained_variance_ratio'] if 'pca_explained_variance_ratio' in m else m['pca_explained_variance']):.0%}).")
    s = lat.sample(min(8000, len(lat)), random_state=0)
    st.plotly_chart(px.scatter_3d(s, x="pc1", y="pc2", z="pc3", color="state", color_discrete_map=COL, category_orders={"state": STATES}, opacity=0.6, height=650), use_container_width=True)

with t3:
    eng = st.selectbox("Engine", sorted(lat.engine_id.unique()))
    e = lat[lat.engine_id == eng].reset_index(drop=True)
    c = st.slider("Cycle", 1, len(e), len(e))
    r = e.iloc[c - 1]
    a, b, d = st.columns(3); a.metric("State", r.state); b.metric("Cycle", int(r.cycle)); d.metric("True RUL (clipped)", int(r.rul))
    st.plotly_chart(px.scatter(e, x="cycle", y="rul", color="state", color_discrete_map=COL, category_orders={"state": STATES}), use_container_width=True)
    st.plotly_chart(px.line_3d(e, x="pc1", y="pc2", z="pc3", color_discrete_sequence=["#888"]), use_container_width=True)
