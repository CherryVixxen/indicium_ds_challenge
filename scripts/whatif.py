import argparse, sys, json
from pathlib import Path
import joblib
import pandas as pd
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import apply_cleaning

def parse_sets(set_list):
    """Converte ['Chave=Valor','Outra=123'] em dict."""
    updates = {}
    for item in set_list or []:
        if "=" not in item:
            raise SystemExit(f"--set inválido: {item}. Use Chave=Valor")
        k, v = item.split("=", 1)
        updates[k.strip()] = v.strip()
    return updates

def predict_one(model_path, record: dict) -> float:
    model = joblib.load(model_path)
    df = pd.DataFrame([record])
    df = apply_cleaning(df)
    X = df.drop(columns=[c for c in ["IMDB_Rating"] if c in df.columns])
    return float(model.predict(X)[0])

def main():
    ap = argparse.ArgumentParser(description="What-if: compara predição antes/depois de mudanças em um registro")
    ap.add_argument("--json", required=True, help="arquivo JSON com 1 registro base")
    ap.add_argument("--model", default="models/imdb_ridge.pkl", help="modelo salvo (.pkl)")
    ap.add_argument("--set", action="append", help='mude campos: ex --set Genre="Drama, Thriller" --set Meta_score=80', dest="sets")
    ap.add_argument("--out", default=None, help="salvar resultado em reports/whatif_*.txt")
    args = ap.parse_args()

    base = json.loads(Path(args.json).read_text(encoding="utf-8"))
    before = predict_one(args.model, base)

    updates = parse_sets(args.sets)
    after_record = {**base, **updates}
    after = predict_one(args.model, after_record)
    delta = after - before

    lines = [
        "WHAT-IF PREDICTION",
        f"modelo: {args.model}",
        f"json:   {args.json}",
        "",
        f"before: {before:.4f}",
        f"after:  {after:.4f}",
        f"delta:  {delta:+.4f}",
        "",
        "changes:",
    ]
    for k, v in updates.items():
        lines.append(f"  {k} = {v}")

    text = "\n".join(lines)
    print(text)

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = f"reports/whatif_{ts}.txt"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"\nsalvo em {args.out}")

if __name__ == "__main__":
    main()
