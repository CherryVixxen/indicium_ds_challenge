import argparse, json, sys
from pathlib import Path
import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import apply_cleaning

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="caminho do arquivo JSON com um registro")
    ap.add_argument("--model", default="models/imdb_ridge.pkl")
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        record = json.load(f)

    df = pd.DataFrame([record])
    df = apply_cleaning(df)
    X = df.drop(columns=[c for c in ["IMDB_Rating"] if c in df.columns])

    model = joblib.load(args.model)
    pred = float(model.predict(X)[0])
    print(f"pred_rating={pred:.4f}")

if __name__ == "__main__":
    main()
