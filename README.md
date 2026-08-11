# 🇧🇷 Brazil Municipal GDP Regression

**A tiny town of 5,465 people in Minas Gerais has a GDP per capita of R$920,828 — nearly 40x the national median, and richer per person than São Paulo, Rio de Janeiro, or Brasília.**

Why? And can a handful of economic indicators — population, sector mix, region — actually predict it?

This project builds a full regression pipeline on real Brazilian government data (IBGE) to predict **GDP per capita for all 5,570 Brazilian municipalities**, walking the complete path a real data science project takes: exploratory analysis → baseline → model comparison with cross-validation → Bayesian hyperparameter tuning with Optuna.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-pipelines-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="XGBoost" src="https://img.shields.io/badge/XGBoost-gradient%20boosting-0A7ABF">
  <img alt="Optuna" src="https://img.shields.io/badge/Optuna-Bayesian%20optimization-6A0DAD">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-serving-009688?logo=fastapi&logoColor=white">
  <img alt="Data source" src="https://img.shields.io/badge/data-IBGE%20(Brazil)-009c3b">
</p>

---

## 🕵️ The mystery behind the data

Look at the top of the GDP-per-capita ranking and something strange jumps out: it's not dominated by big cities. It's dominated by **small towns sitting on top of mining and oil operations**.

| Municipality | State | Population | GDP per capita (R$) |
|---|---|---:|---:|
| Catas Altas | MG | 5,465 | **920,828** |
| Canaã dos Carajás | PA | 39,103 | 894,763 |
| São Gonçalo do Rio Abaixo | MG | 11,114 | 684,163 |
| Itatiaiuçu | MG | 11,354 | 610,742 |
| Presidente Kennedy | ES | 11,741 | 580,174 |

For scale: the **national median is R$23,373**. These municipalities are 20-40x above it — because a small population divides a huge, concentrated industrial GDP (iron ore, oil royalties) into an enormous per-capita figure. It's a textbook case of a right-skewed target variable (skewness ≈ 7.9) that quietly breaks naive models and naive intuitions alike.

There's a second surprise buried in the regional averages — the kind you only find by actually plotting the data instead of assuming it:

| Region | Median GDP per capita (R$) |
|---|---:|
| **South** | **43,022** |
| Center-West | 38,486 |
| **Southeast** | 25,312 |
| North | 18,602 |
| Northeast | 11,499 |

The Southeast — home to São Paulo and Rio de Janeiro, Brazil's economic giants — isn't #1. The **South** is, driven by a broad base of mid-sized industrial and agribusiness municipalities rather than a few mega-cities. Averages hide as much as they reveal, and this project's [EDA notebook](notebooks/01_EDA.ipynb) exists to surface exactly this kind of thing before a single model gets trained.

---

## 📊 TL;DR — does it actually work?

| Stage | Model | RMSE (R$) | MAE (R$) | R² |
|---|---|---:|---:|---:|
| Baseline | Linear Regression | 33,355 | 14,863 | 0.382 |
| Baseline | Random Forest | 17,253 | 3,921 | 0.835 |
| 5-fold CV (train) | XGBoost | 13,124 ± 2,694 | 3,972 | 0.896 ± 0.035 |
| **Optuna-tuned (final, held-out test)** | **XGBoost** | 22,854 | **3,944** | 0.710 |

A plain linear model barely explains a third of the variance (R² = 0.38) — no surprise, given how non-linear the relationship between sector composition and prosperity really is. Tree-based models close that gap dramatically, and **XGBoost consistently wins across baseline, cross-validation, and Bayesian tuning.**

**Worth calling out honestly:** the tuned model's cross-validated RMSE (R$13.1k) looks a lot better than its final test-set RMSE (R$22.9k) — while MAE barely moves (R$3.97k → R$3.94k). That gap is the skewed target coming back to bite: a handful of mining-town outliers landed in the test split and, by chance, disproportionately inflate a squared-error metric. It's exactly the kind of thing this README opened with — and exactly why the [EDA notebook](notebooks/01_EDA.ipynb) flags `log1p`-transforming the target as a promising next experiment. A project that hides this would look better on paper and teach you less.

---

## 🧭 How the project is organized

```
Regressao/
├── data/                      # raw + processed data (gitignored)
├── src/
│   ├── baixar_dados.py        # pulls & builds the dataset straight from the IBGE API
│   └── api.py                 # FastAPI service that serves live predictions from the trained model
├── notebooks/
│   ├── 01_EDA.ipynb                    # exploratory analysis only — no modeling
│   ├── 02_Baseline.ipynb               # train/test split, pipeline, first models
│   ├── 03_Comparacao_Modelos_CV.ipynb  # Linear/Ridge/RandomForest/XGBoost, 5-fold CV
│   └── 04_Otimizacao_Optuna.ipynb      # Bayesian hyperparameter search + final evaluation
├── models/                    # trained pipelines (.joblib, gitignored)
├── reports/                   # exported charts (gitignored)
└── requirements.txt
```

Each notebook is a self-contained checkpoint — read them in order and you're literally re-living the project's decisions, not just its code.

## 🧬 Where the data comes from

Everything here is public, official Brazilian government data — no scraping, no shortcuts:

- **[IBGE SIDRA — table 5938](https://sidra.ibge.gov.br/tabela/5938)**: GDP of Brazilian municipalities, broken down by economic sector (agriculture, industry, services, public administration).
- **[IBGE SIDRA — table 6579](https://sidra.ibge.gov.br/tabela/6579)**: estimated resident population.

[`src/baixar_dados.py`](src/baixar_dados.py) hits the IBGE API directly, merges both sources for all 5,570 municipalities, and derives the target (`pib_per_capita`). Re-running it always regenerates `data/pib_municipios.csv` from scratch — the dataset is fully reproducible, not a static file someone hand-curated.

**Features → target:**

| Feature | What it captures |
|---|---|
| `populacao` | Estimated population |
| `participacao_agropecuaria` | % of local GDP from agriculture |
| `participacao_industria` | % of local GDP from industry |
| `participacao_servicos` | % of local GDP from services |
| `participacao_administracao_publica` | % of local GDP from public administration |
| `uf`, `regiao` | State and macro-region |
| → `pib_per_capita` | **Target**: GDP per capita (R$) |

Note the absolute GDP value itself is deliberately *not* a feature — only its sector composition is. Including it would let the target be reconstructed by simple arithmetic, which would make the "prediction" trivial and pointless.

## 🔬 The pipeline, notebook by notebook

1. **[EDA](notebooks/01_EDA.ipynb)** — distribution of the target, outlier hunting, regional breakdowns, correlations. Sets up every modeling decision that follows.
2. **[Baseline](notebooks/02_Baseline.ipynb)** — train/test split *before* any transformation (no leakage), a `ColumnTransformer` pipeline, and two reference models.
3. **[Model comparison + cross-validation](notebooks/03_Comparacao_Modelos_CV.ipynb)** — Linear Regression, Ridge, Random Forest, and XGBoost compared with 5-fold CV on the training set only, visualized side by side.
4. **[Bayesian optimization with Optuna](notebooks/04_Otimizacao_Optuna.ipynb)** — a TPE sampler searches the hyperparameter space of both tree-based models, then the champion is evaluated **once** on the untouched test set.

## 🌐 Serving predictions with FastAPI

The model doesn't just live in a notebook — [`src/api.py`](src/api.py) wraps the saved pipeline (`models/modelo_final.joblib`) in a small FastAPI service, so you can query it like any other backend.

```bash
uvicorn src.api:app --reload
```

That starts the API at `http://127.0.0.1:8000`, with interactive docs (Swagger) at `/docs`. A single `POST /prever` call takes a municipality's raw indicators and returns its predicted GDP per capita:

```bash
curl -X POST http://127.0.0.1:8000/prever \
  -H "Content-Type: application/json" \
  -d '{
    "populacao": 111148,
    "participacao_agropecuaria": 10.5,
    "participacao_industria": 14.61,
    "participacao_servicos": 46.86,
    "participacao_administracao_publica": 28.03,
    "uf": "RO",
    "regiao": "Norte"
  }'
# {"pib_per_capita_previsto": 28385.96, ...}
```

For reference, that's Ariquemes-RO — its real GDP per capita is R$28,892, so the model landed within ~1.7% of it. Before the request even reaches the model, Pydantic validators reject anything that couldn't be a real municipality: an unknown state/region code, or sector shares that don't add up to roughly 100%.

## 🚀 Run it yourself

```bash
# 1. activate the virtual environment
.\venv\Scripts\Activate.ps1        # Windows
source venv/bin/activate           # macOS/Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. (re)build the dataset straight from the IBGE API
cd src && python baixar_dados.py

# 4. open the notebooks, in order
jupyter lab notebooks/

# 5. optional — once 04_Otimizacao_Optuna.ipynb has produced models/modelo_final.joblib,
#    serve live predictions:
uvicorn src.api:app --reload
```

## 🛠️ Stack

`pandas` · `numpy` · `scikit-learn` · `xgboost` · `optuna` · `matplotlib` · `seaborn` · `joblib` · `fastapi` · `uvicorn`

## 🔭 What's next

- [ ] Log-transform the target to tame the extreme mining/oil-town outliers
- [ ] Feature: distance to the nearest state capital or metropolitan region
- [ ] SHAP values to explain *why* the model predicts what it predicts, municipality by municipality
- [ ] A small Streamlit front-end on top of the FastAPI service, so anyone can try a prediction without touching `curl`

If any of that sounds interesting, watch this repo — or better yet, open a PR.
