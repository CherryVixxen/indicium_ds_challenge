# Entregas

## 1- Faça uma análise exploratória dos dados (EDA), demonstrando as principais características entre as variáveis e apresentando algumas hipóteses relacionadas. Seja criativo!
    - Notas encontradas entre 7.6 e 8.4 aproximadamente, e cauda curta acima de 9

    - "No_of_Votes" cresce com a nota, a qualidade maior acompanha uma popularidade maior

    - "Meta_score" correlaciona positivamente com "IMDB_Rating"

    - Gênero Drama, domina em volume (Ranking em reports/genres_stats.csv)

    Hipótese 1: Quanto mais pessoas votam maior a chance de uma nota mais alta

    Hipótese 2: "Meta_score" é um bom preditor de nota

    Hipótese 3: Gêneros dominantes concentram notas altas por amostragem

## 2 - Responda também às seguintes perguntas:
### a) Qual filme você recomendaria para uma pessoa que você não conhece?
    The Godfather — nota 9.2, votos 1620367
### b) Quais são os principais fatores que estão relacionados com alta expectativa de faturamento de um filme? 
    Em pré lançamento: combinações populares de gênero(s), resumo, direção, elenco, duração, lançamento e certificado.
 ### c) Quais insights podem ser tirados com a coluna Overview? É possível inferir o gênero do filme a partir dessa coluna?
    O overview é um resumo curto que ajuda a ter uma idea de tema e tom do filme. Termos do resumo discriminam bem temas como “war”, “heist”, “romance”, podendo assim, inferir gênero a partir dessa coluna.

## 3 - Explique como você faria a previsão da nota do imdb a partir dos dados. Quais variáveis e/ou suas transformações você utilizou e por quê? Qual tipo de problema estamos resolvendo (regressão, classificação)? Qual modelo melhor se aproxima dos dados e quais seus prós e contras? Qual medida de performance do modelo foi escolhida e por quê?
    Para prever a nota do IMDb, tratei o problema como regressão com alvo contínuo (IMDB_Rating). Usei um conjunto de variáveis que captura sinal de época/mercado, crítica, popularidade, conteúdo e “efeito nome”: numéricas (Released_Year, Meta_score, No_of_Votes, Runtime_min, Gross_usd), categóricas (Certificate, Genre, Director, Star1–Star4) e texto (Overview). Na limpeza, removi índices antigos, extraí a duração em minutos a partir de Runtime e normalizei Gross para numérico; depois apliquei imputação (mediana/moda), padronização nos numéricos, one-hot nas categóricas e TF-IDF no Overview. Tudo roda dentro de um Pipeline com ColumnTransformer, evitando vazamento de dados, pois o fit acontece apenas nos dados de treino em cada partição. O modelo escolhido foi Ridge (regressão linear com L2), que funciona bem com a matriz esparsa gerada por TF-IDF + one-hot, é rápido e interpretável via coeficientes; como contrapartida, não captura tão bem não linearidades e interações complexas. A validação combina hold-out 80/20 e K-Fold (5) para estabilidade; as métricas são RMSE (principal, por punir mais erros grandes) e MAE (complementar). No CSV do desafio, o baseline com Ridge ficou em RMSE ≈ 0,20 e MAE ≈ 0,16, valores reproduzíveis pelos scripts do repositório e pelo modelo salvo em .pkl.
    
## 4 - Qual seria a nota do IMDB supondo um filme com as seguintes características:
##### {'Series_Title': 'The Shawshank Redemption',
##### 'Released_Year': '1994',
##### 'Certificate': 'A',
##### 'Runtime': '142 min',
##### 'Genre': 'Drama',
##### 'Overview': 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
##### 'Meta_score': 80.0,
##### 'Director': 'Frank Darabont',
##### 'Star1': 'Tim Robbins',
##### 'Star2': 'Morgan Freeman',
##### 'Star3': 'Bob Gunton',
##### 'Star4': 'William Sadler',
##### 'No_of_Votes': 2343110,
##### 'Gross': '28,341,469'}
    pred_rating=9.4418

## 5 - Salve o modelo desenvolvido no formato .pkl. 
    models/imdb_ridge.pkl (gerado por scripts/train.py ou pelo notebook).

## 6 - A entrega deve ser feita através de um repositório de código público que contenha:
### a) README explicando como instalar e executar o projeto
    README.md com comandos de treino, predição e avaliação.
### b) Arquivo de requisitos com todos os pacotes utilizados e suas versões
    requirements.txt (numpy, pandas, scikit-learn, joblib com versões).
### c) Relatórios das análises estatísticas e EDA em PDF, Jupyter Notebook ou semelhante conforme passo 1 e 2.
    notebooks/01_eda.ipynb e notebooks/02_eda.ipynb + figuras em reports/.
### d) Códigos de modelagem utilizados no passo 3 (pode ser entregue no mesmo Jupyter Notebook).
    src/ (data, features, model) e scripts/ (train, predict, evaluate, crossval, whatif).
### e) Arquivo .pkl conforme passo 5 acima.
    models/imdb_ridge.pkl.

## Todos os códigos produzidos devem seguir as boas práticas de codificação.
    Fiz o que pude, não pude muito.