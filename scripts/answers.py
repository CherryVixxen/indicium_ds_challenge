from pathlib import Path
import sys, numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_clean

def main():
    df = load_clean()

    # 1) Relação entre votos e nota (correlação de Pearson)
    votes_corr = df["No_of_Votes"].astype(float).corr(df["IMDB_Rating"].astype(float))

    # 2) Diretores com mais de N filmes e melhor média de nota
    g_dir = (
        df.groupby("Director", dropna=False)["IMDB_Rating"]
          .agg(["count", "mean"])
          .rename(columns={"count": "n_filmes", "mean": "media"})
          .sort_values(["n_filmes", "media"], ascending=[False, False])
    )
    top_dirs = g_dir[g_dir["n_filmes"] >= 3].head(10)

    # 3) Gêneros com melhor média (tratando múltiplos gêneros por filme)
    genres = (
        df.assign(Genre=df["Genre"].fillna("").astype(str).str.split(","))
          .explode("Genre")
    )
    genres["Genre"] = genres["Genre"].str.strip()
    g_gen = (
        genres[genres["Genre"] != ""]
        .groupby("Genre")["IMDB_Rating"]
        .agg(["count", "mean"])
        .rename(columns={"count": "n_titulos", "mean": "media"})
        .sort_values(["n_titulos", "media"], ascending=[False, False])
    )
    top_gen = g_gen[g_gen["n_titulos"] >= 10].head(10)

    out = Path("reports/answers.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("Respostas iniciais (baseline)\n")
        f.write("\n1) Correlação No_of_Votes x IMDB_Rating\n")
        f.write(f"pearson={votes_corr:.4f}\n")

        f.write("\n2) Diretores com >=3 filmes (top 10 por n_filmes e média)\n")
        f.write(top_dirs.round(3).to_string() + "\n")

        f.write("\n3) Gêneros com >=10 títulos (top 10 por n_titulos e média)\n")
        f.write(top_gen.round(3).to_string() + "\n")

    print(f"ok: salvo em {out}")

if __name__ == "__main__":
    main()
