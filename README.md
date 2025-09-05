# Desafio de Ciência de Dados da Lighthouse
#### Setembro de 2025

## Como Executar

#### 1) Treinar e salvar modelo

    python scripts/train.py --out models/imdb_ridge.pkl


#### 2) Prever em lote a partir do CSV em data/raw 
    
    python scripts/predict_csv.py --model models/imdb_ridge.pkl --out data/processed/preds_baseline.csv


#### 3) Prever um registro único a partir de JSON

    python scripts/predict_one.py --json examples/example_one.json --model models/imdb_ridge.pkl


#### 4) Avaliar no dataset completo

    python scripts/evaluate.py --model models/imdb_ridge.pkl --out reports/metrics.json


#### 5) Validação cruzada

    python scripts/crossval.py --folds 5 --out reports/cv.json

