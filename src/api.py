# API para servir previsoes do modelo de PIB per capita municipal.
#
# O modelo servido aqui e o pipeline completo (pre-processador + XGBoost otimizado com Optuna)
# treinado e salvo em ../models/modelo_final.joblib pelo notebook notebooks/04_Otimizacao_Optuna.ipynb.
# "Pipeline completo" significa que o StandardScaler/OneHotEncoder do pre-processamento ja estao
# dentro do objeto salvo — quem chama esta API nao precisa saber como o modelo foi treinado, so
# precisa mandar os dados brutos do municipio.
#
# Como rodar (a partir da raiz do projeto, com o venv ativado):
#   uvicorn src.api:app --reload
#
# Depois, a documentacao interativa (Swagger) fica disponivel em http://127.0.0.1:8000/docs

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Constantes do dominio
#
# Sao as mesmas 27 UFs e 5 regioes que aparecem em data/pib_municipios.csv. Validar contra essas
# listas aqui, na entrada da API, evita que um erro de digitacao (ex.: uf="MGG") va parar
# silenciosamente no OneHotEncoder do pipeline, que trataria como categoria desconhecida
# (handle_unknown='ignore') e devolveria uma previsao sem nenhum aviso.
# ---------------------------------------------------------------------------
UFS_VALIDAS = {
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT",
    "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
}
REGIOES_VALIDAS = {"Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"}

# caminho do modelo relativo a este arquivo, nao ao diretorio de onde o uvicorn e chamado —
# assim a API funciona tanto rodando "uvicorn src.api:app" da raiz quanto "uvicorn api:app" de dentro de src/
CAMINHO_MODELO = Path(__file__).resolve().parent.parent / "models" / "modelo_final.joblib"

# guarda o pipeline carregado; comeca como None e so e preenchido no startup da API (ver lifespan
# abaixo). Usar uma variavel de modulo simples é suficiente aqui porque so existe UM modelo servido.
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Codigo executado UMA VEZ quando a API sobe — nao a cada requisicao.

    Carregar o modelo aqui (em vez de dentro do endpoint) evita reabrir o arquivo .joblib a cada
    chamada, o que seria lento e desnecessario, ja que o pipeline nao muda entre requisicoes.
    """
    global pipeline

    if not CAMINHO_MODELO.exists():
        raise RuntimeError(
            f"Modelo nao encontrado em {CAMINHO_MODELO}. "
            "Rode o notebook notebooks/04_Otimizacao_Optuna.ipynb ate o fim para gera-lo."
        )

    pipeline = joblib.load(CAMINHO_MODELO)
    yield  # a API fica no ar aqui; o codigo depois do yield rodaria no desligamento (nao precisamos de nada)


app = FastAPI(
    title="API de Previsao de PIB per Capita Municipal",
    description=(
        "Serve previsoes do pipeline treinado em 04_Otimizacao_Optuna.ipynb "
        "(pre-processamento + XGBoost otimizado com Optuna) sobre dados publicos do IBGE."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas (formato dos dados que entram e saem da API)
#
# Usar Pydantic aqui faz a validacao acontecer ANTES do request chegar na funcao do endpoint:
# tipo errado, valor fora do intervalo ou UF invalida ja voltam como erro 422 com uma mensagem
# clara, sem precisar escrever "if" de validacao manualmente dentro da view.
# ---------------------------------------------------------------------------
class MunicipioEntrada(BaseModel):
    populacao: int = Field(
        ..., gt=0, description="Populacao estimada do municipio (pessoas)", examples=[50000]
    )
    participacao_agropecuaria: float = Field(
        ..., ge=0, le=100, description="% do VAB municipal vindo da agropecuaria", examples=[15.0]
    )
    participacao_industria: float = Field(
        ..., ge=0, le=100, description="% do VAB municipal vindo da industria", examples=[20.0]
    )
    participacao_servicos: float = Field(
        ...,
        ge=0,
        le=100,
        description="% do VAB municipal vindo de servicos (excl. administracao publica)",
        examples=[45.0],
    )
    participacao_administracao_publica: float = Field(
        ...,
        ge=0,
        le=100,
        description="% do VAB municipal vindo da administracao publica, defesa, educacao e saude publicas",
        examples=[20.0],
    )
    uf: str = Field(..., description="Sigla da UF, maiuscula (ex.: 'MG')", examples=["MG"])
    regiao: str = Field(..., description="Grande regiao do Brasil (ex.: 'Sudeste')", examples=["Sudeste"])

    @field_validator("uf")
    @classmethod
    def validar_uf(cls, valor: str) -> str:
        valor = valor.strip().upper()
        if valor not in UFS_VALIDAS:
            raise ValueError(f"UF '{valor}' invalida. Use uma das 27 siglas: {sorted(UFS_VALIDAS)}")
        return valor

    @field_validator("regiao")
    @classmethod
    def validar_regiao(cls, valor: str) -> str:
        valor = valor.strip().title()
        if valor not in REGIOES_VALIDAS:
            raise ValueError(f"Regiao '{valor}' invalida. Use uma das 5: {sorted(REGIOES_VALIDAS)}")
        return valor

    @model_validator(mode="after")
    def validar_soma_das_participacoes(self) -> "MunicipioEntrada":
        # as 4 participacoes setoriais representam fatias do mesmo bolo (o VAB total do municipio),
        # entao devem somar ~100%. Uma tolerancia de 1 ponto percentual cobre arredondamento normal
        # sem deixar passar um erro grosseiro de digitacao (ex.: esquecer de dividir por 10).
        soma = (
            self.participacao_agropecuaria
            + self.participacao_industria
            + self.participacao_servicos
            + self.participacao_administracao_publica
        )
        if abs(soma - 100) > 1:
            raise ValueError(
                f"As participacoes setoriais devem somar ~100% (soma atual: {soma:.1f}%). "
                "Confira os quatro valores."
            )
        return self


class PrevisaoSaida(BaseModel):
    pib_per_capita_previsto: float = Field(..., description="PIB per capita previsto, em reais (R$)")
    entrada_validada: MunicipioEntrada = Field(
        ..., description="Os dados de entrada apos validacao/normalizacao (uf e regiao padronizadas)"
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", summary="Informacoes basicas da API")
def raiz() -> dict:
    return {
        "aplicacao": app.title,
        "versao": app.version,
        "documentacao": "/docs",
        "endpoint_de_previsao": "/prever (POST)",
    }


@app.get("/saude", summary="Verifica se a API subiu e o modelo foi carregado")
def saude() -> dict:
    return {"status": "ok", "modelo_carregado": pipeline is not None}


@app.post(
    "/prever",
    response_model=PrevisaoSaida,
    summary="Preve o PIB per capita de um municipio a partir dos seus indicadores",
)
def prever_pib_per_capita(dados: MunicipioEntrada) -> PrevisaoSaida:
    if pipeline is None:
        # so aconteceria se o lifespan nao tivesse rodado (ex.: uvicorn chamado de forma nao-padrao)
        raise HTTPException(status_code=503, detail="Modelo ainda nao foi carregado pela API.")

    # o pipeline foi treinado esperando um DataFrame com essas colunas (mesmos nomes usados em
    # FEATURES_NUM/FEATURES_CAT nos notebooks). O ColumnTransformer localiza as colunas pelo NOME,
    # entao a ordem das chaves do dicionario abaixo nao importa — so os nomes precisam bater.
    entrada_df = pd.DataFrame(
        [
            {
                "populacao": dados.populacao,
                "participacao_agropecuaria": dados.participacao_agropecuaria,
                "participacao_industria": dados.participacao_industria,
                "participacao_servicos": dados.participacao_servicos,
                "participacao_administracao_publica": dados.participacao_administracao_publica,
                "uf": dados.uf,
                "regiao": dados.regiao,
            }
        ]
    )

    # predict() devolve um array (mesmo para uma linha só); pegamos o primeiro (e único) elemento
    previsao = float(pipeline.predict(entrada_df)[0])

    return PrevisaoSaida(
        pib_per_capita_previsto=round(previsao, 2),
        entrada_validada=dados,
    )


# ---------------------------------------------------------------------------
# Permite rodar com "python src/api.py" (alem do jeito recomendado "uvicorn src.api:app --reload"),
# util para quem preferir clicar em "Run" no editor em vez de usar o terminal.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
