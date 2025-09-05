import argparse, sys, json
from pathlib import Path
import numpy as np
import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_clean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/imdb_ridge.pkl")
    ap.add_argument("--csv", help="opcional: caminho de CSV; padrão = data/raw/<primeiro .csv>")
    ap.add_argument("--out", default="reports/metrics.json")
    args = ap.parse_args()

    df = load_clean(args.csv) if args.csv else load_clean()
    if "IMDB_Rating" not in df.columns:
        raise SystemExit("Dataset sem coluna IMDB_Rating para avaliar.")

    X = df.drop(columns=["IMDB_Rating"])
    y = df["IMDB_Rating"].astype(float).to_numpy()

    model = joblib.load(args.model)
    pred = model.predict(X)
    rmse = float(np.sqrt(((y - pred) ** 2).mean()))
    mae  = float(np.abs(y - pred).mean())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"rmse": rmse, "mae": mae, "n": int(len(y))}, f, ensure_ascii=False, indent=2)

    print(f"RMSE={rmse:.4f}  MAE={mae:.4f}  n={len(y)}")
    print(f"salvo em {args.out}")

if __name__ == "__main__":
    main()
