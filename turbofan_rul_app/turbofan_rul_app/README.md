# Latent Engine Health States (C-MAPSS FD001)

1. `pip install -r requirements.txt`
2. Download NASA C-MAPSS and put `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` in `data/raw/`.
3. `python -m src.train` -> writes `artifacts/` (metrics.json + parquet files, small, commit them).
4. `pytest tests/` then `streamlit run app.py`.

## Deploy
- **Streamlit Community Cloud:** push repo (with `artifacts/`), set main file to `app.py`.
- **Docker / Cloud Run / Fly / Render:** `docker build -t rul . && docker run -p 8080:8080 rul`.
The app only reads `artifacts/`, so it starts instantly and needs no data or training in production.

Protocol: scaler/PCA/k-means fit on train only; clusters ordered post-hoc by train RUL; test = last cycle per engine; bootstrap CI over engines; random-label control included.
