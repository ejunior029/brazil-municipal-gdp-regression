# Baixa e monta o dataset de PIB per capita dos municipios brasileiros
# Fonte: IBGE - Sistema IBGE de Recuperacao Automatica (API SIDRA / servicodados.ibge.gov.br)
#   Tabela 5938: Produto Interno Bruto dos Municipios (ano de referencia: ver ANO abaixo)
#   Tabela 6579: Populacao residente estimada (ano de referencia: ver ANO abaixo)
# Portal: https://www.ibge.gov.br/estatisticas/economicas/contas-nacionais/9088-produto-interno-bruto-dos-municipios.html
import requests
import pandas as pd

ANO = "2021"  # 2022/2023 ainda nao tem a abertura por setor (agropecuaria/industria/servicos) publicada por municipio
BASE = "https://servicodados.ibge.gov.br/api/v3/agregados"

# variaveis da tabela 5938 (valores em % de participacao no VAB total do municipio)
VARIAVEIS_PIB = {
    "516": "participacao_agropecuaria",
    "520": "participacao_industria",
    "6574": "participacao_servicos",
    "528": "participacao_administracao_publica",
}
VARIAVEL_POPULACAO = "9324"  # tabela 6579 - populacao residente estimada
VARIAVEL_PIB_TOTAL = "37"  # tabela 5938 - PIB a precos correntes (Mil Reais) - usado so p/ calcular o alvo


def buscar_variavel(tabela, variavel, ano):
    """Busca UMA variavel do SIDRA/IBGE para TODOS os municipios do Brasil, num unico ano.

    Monta a URL do endpoint de agregados do IBGE (docs: https://servicodados.ibge.gov.br/api/docs/agregados)
    para a tabela e variavel pedidas, no nivel territorial N6 (municipio), com "localidades=N6[all]"
    trazendo os ~5.570 municipios numa unica chamada.

    Parametros
    ----------
    tabela : str
        Codigo da tabela SIDRA (ex.: "5938" para PIB dos Municipios, "6579" para populacao).
    variavel : str
        Codigo da variavel dentro da tabela (ex.: "37" = PIB a precos correntes). Cada tabela do
        SIDRA tem seu proprio conjunto de variaveis — ver VARIAVEIS_PIB e VARIAVEL_POPULACAO acima
        para os codigos usados neste script.
    ano : str
        Ano de referencia (ex.: "2021"). Precisa ser um ano em que a API tenha esse dado publicado
        para essa tabela/variavel — nem toda combinacao existe para todo ano (ver comentario da
        constante ANO).

    Retorna
    -------
    pandas.DataFrame
        Uma linha por municipio, com as colunas:
        - codigo_municipio: codigo IBGE do municipio (chave usada para os merges em main())
        - municipio_uf: nome no formato "Municipio - UF", ainda nao separado
        - valor: valor da variavel para aquele municipio naquele ano (ainda como string — a API
          do IBGE devolve numeros como texto, e usa "-" ou ".." para indicar dado indisponivel)
    """
    url = f"{BASE}/{tabela}/periodos/{ano}/variaveis/{variavel}?localidades=N6[all]"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
    resp.raise_for_status()
    # a resposta tem um nivel de aninhamento para suportar variaveis cruzadas com classificacoes
    # (ex.: "por sexo", "por faixa etaria"); nenhuma das variaveis usadas aqui tem esse cruzamento,
    # entao sempre pegamos o primeiro (e unico) resultado
    dados = resp.json()[0]["resultados"][0]["series"]
    linhas = []
    for item in dados:
        loc = item["localidade"]
        valor = item["serie"].get(ano)
        linhas.append({"codigo_municipio": loc["id"], "municipio_uf": loc["nome"], "valor": valor})
    return pd.DataFrame(linhas)


def main():
    """Monta o dataset final (data/pib_municipios.csv) a partir de varias chamadas a buscar_variavel().

    Passo a passo:
    1. Baixa o PIB total de cada municipio (usado so para calcular o alvo, nao entra como feature).
    2. Baixa a populacao estimada de cada municipio.
    3. Baixa as 4 participacoes setoriais no VAB municipal (agropecuaria, industria, servicos,
       administracao publica) — essas sim entram como features do modelo.
    4. Junta tudo num unico DataFrame pelo codigo_municipio (chave que nao muda entre tabelas).
    5. Separa "Municipio - UF" em duas colunas, e deriva a grande regiao a partir da UF.
    6. Converte as colunas numericas de string para float (a API do IBGE devolve tudo como texto).
    7. Calcula o alvo: pib_per_capita = pib_total_mil_reais * 1000 / populacao.
    8. Descarta linhas sem alvo ou sem populacao (municipios com dado faltante na fonte) e salva
       o CSV final em ../data/pib_municipios.csv, so com as colunas que os notebooks usam.

    Nao recebe parametros nem retorna valor — roda como um script (ver `if __name__ == "__main__"`
    no fim do arquivo) e o resultado e o efeito colateral de escrever o CSV em disco.
    """
    print("Baixando PIB total (para calcular o alvo)...")
    df_pib = buscar_variavel("5938", VARIAVEL_PIB_TOTAL, ANO).rename(columns={"valor": "pib_total_mil_reais"})

    print("Baixando populacao estimada...")
    df_pop = buscar_variavel("6579", VARIAVEL_POPULACAO, ANO).rename(columns={"valor": "populacao"})

    df = df_pib.merge(df_pop[["codigo_municipio", "populacao"]], on="codigo_municipio", how="left")

    for var_id, nome_coluna in VARIAVEIS_PIB.items():
        print(f"Baixando {nome_coluna}...")
        df_var = buscar_variavel("5938", var_id, ANO).rename(columns={"valor": nome_coluna})
        df = df.merge(df_var[["codigo_municipio", nome_coluna]], on="codigo_municipio", how="left")

    # separa nome do municipio e UF (formato "Municipio - UF")
    partes = df["municipio_uf"].str.rsplit(" - ", n=1, expand=True)
    df["municipio"] = partes[0]
    df["uf"] = partes[1]

    regiao_por_uf = {
        "RO": "Norte", "AC": "Norte", "AM": "Norte", "RR": "Norte", "PA": "Norte", "AP": "Norte", "TO": "Norte",
        "MA": "Nordeste", "PI": "Nordeste", "CE": "Nordeste", "RN": "Nordeste", "PB": "Nordeste",
        "PE": "Nordeste", "AL": "Nordeste", "SE": "Nordeste", "BA": "Nordeste",
        "MG": "Sudeste", "ES": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
        "PR": "Sul", "SC": "Sul", "RS": "Sul",
        "MS": "Centro-Oeste", "MT": "Centro-Oeste", "GO": "Centro-Oeste", "DF": "Centro-Oeste",
    }
    df["regiao"] = df["uf"].map(regiao_por_uf)

    # converte colunas numericas (a API retorna string; "-" e ".." viram NaN)
    colunas_numericas = ["pib_total_mil_reais", "populacao"] + list(VARIAVEIS_PIB.values())
    for col in colunas_numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # alvo: PIB per capita em reais
    df["pib_per_capita"] = (df["pib_total_mil_reais"] * 1000) / df["populacao"]

    colunas_finais = [
        "codigo_municipio", "municipio", "uf", "regiao", "populacao",
        "participacao_agropecuaria", "participacao_industria",
        "participacao_servicos", "participacao_administracao_publica",
        "pib_per_capita",
    ]
    df = df[colunas_finais].dropna(subset=["pib_per_capita", "populacao"])

    caminho_saida = "../data/pib_municipios.csv"
    df.to_csv(caminho_saida, index=False)
    print(f"Salvo em {caminho_saida} — {df.shape[0]} linhas, {df.shape[1]} colunas")


if __name__ == "__main__":
    main()
