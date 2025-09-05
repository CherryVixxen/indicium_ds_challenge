from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def raw_dir(root: Path | None = None) -> Path:
    base = Path(root) if root else ROOT
    return base / "data" / "raw"

def find_csv(root: Path | None = None) -> Path:
    files = sorted(raw_dir(root).glob("*.csv"))
    if not files:
        raise FileNotFoundError("Nenhum CSV em data/raw")
    return files[0]

def load_clean(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else find_csv()
    df = pd.read_csv(p)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    for col in ["Released_Year", "Meta_score", "No_of_Votes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Runtime_min"] = df["Runtime"].astype(str).str.extract(r"(\d+)")[0].astype(float)

    tmp = df["Gross"].astype(str).str.replace(r"[,\$]", "", regex=True)
    tmp = tmp.where(tmp.str.lower() != "nan", np.nan)
    df["Gross_usd"] = pd.to_numeric(tmp, errors="coerce")

    return df