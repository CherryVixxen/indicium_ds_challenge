import argparse, sys
from pathlib import Path
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_clean
import src.model as mdl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/imdb_ridge.pkl", help="caminho do .pkl de saída")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_clean()
    rmse, mae, pipe = mdl.train_validate(df, test_size=args.test_size, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.out)

    print(f"RMSE={rmse:.4f}  MAE={mae:.4f}")
    print(f"modelo salvo em: {args.out}")

if __name__ == "__main__":
    main()