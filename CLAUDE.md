# Projeto: Regressao

## Objetivo
Prever um valor numerico continuo a partir dos dados de entrada.

## Stack
- Python 3, pandas, numpy, scikit-learn, matplotlib, seaborn
- Ambiente virtual em ./venv

## Estrutura de pastas
- data/       -> dados (nao versionados)
- notebooks/  -> EDA e experimentos
- src/        -> codigo reutilizavel
- models/     -> modelos salvos (.pkl / .joblib)
- reports/    -> graficos e resultados

## Algoritmos previstos
LinearRegression, RandomForestRegressor, XGBoostRegressor

## Metrica principal
RMSE, MAE, R2

## Convencoes
- Comentar o codigo em portugues
- SEMPRE separar treino/teste ANTES de qualquer transformacao (evitar data leakage)
- Salvar graficos em ./reports/
- Salvar modelos treinados em ./models/
- Trabalhar em passos pequenos: EDA -> baseline -> modelos -> tuning -> avaliacao
