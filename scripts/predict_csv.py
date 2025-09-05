import argparse
from pathlib import Path
import sys
import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_clean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV para prever; se não passar, usa o de data/raw")
    ap.add_argument("--model", default="models/imdb_ridge.pkl")
    ap.add_argument("--out", default="data/processed/predictions.csv")
    args = ap.parse_args()

    df = load_clean(args.csv) if args.csv else load_clean()
    X = df.drop(columns=[c for c in ["IMDB_Rating"] if c in df.columns])

    model = joblib.load(args.model)
    preds = model.predict(X)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "Series_Title": df["Series_Title"] if "Series_Title" in df.columns else range(len(df)),
        "pred_rating": preds
    }).to_csv(args.out, index=False)

    print(f"ok: {len(preds)} previsões salvas em {args.out}")

if __name__ == "__main__":
    main()