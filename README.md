# Desafio de Ciência de Dados da Lighthouse
#### Setembro de 2025

## Como Instalar

### Windows (PowerShell)

#### 1) Criar e ativar o ambiente virtual
    python -m venv .venv
    . .\.venv\Scripts\Activate.ps1

#### 2) Atualizar o pip
    python -m pip install --upgrade pip

#### 3) Instalar as dependências
    pip install -r requirements.txt

#### 4) Selecionar a .venv no VS Code (opcional)
    Ctrl+Shift+P → Python: Select Interpreter → escolha a .venv do projeto

#### 5) Teste rápido
    python -c "import pandas, sklearn, joblib; print('ok')"


### macOS / Linux

#### 1) Criar e ativar o ambiente virtual
    python3 -m venv .venv
    source .venv/bin/activate

#### 2) Atualizar o pip
    python -m pip install --upgrade pip

#### 3) Instalar as dependências
    pip install -r requirements.txt

#### 4) Selecionar a .venv no VS Code (opcional)
    Ctrl+Shift+P → Python: Select Interpreter → escolha a .venv do projeto

#### 5) Teste rápido
    python -c "import pandas, sklearn, joblib; print('ok')"

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

