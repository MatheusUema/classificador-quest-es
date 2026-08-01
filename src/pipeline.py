"""
Pipeline de pré-processamento do ENEM.
Extrai questões dos PDFs P1 AMARELO, faz join com parâmetros IRT
dos microdados INEP e gera dataset com features de texto + difficulty_score.
"""

import re
import sys
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ANOS = [2020, 2021, 2022, 2023, 2024]

DAY1_AREAS = {"LC", "CH"}
DAY2_AREAS = {"CN", "MT"}

AREA_NAMES = {
    "LC": "area_linguagens",
    "CH": "area_humanas",
    "CN": "area_natureza",
    "MT": "area_matematica",
}

# Ordem canônica das features — usada por train.py e router.py
FEATURES = [
    "n_palavras",
    "media_palavras_por_sentenca",
    "pct_palavras_longas",
    "n_conectivos_logicos",
    "tem_negacao",
    "tem_formula_ou_numero",
    "tem_imagem",
    "area_linguagens",
    "area_humanas",
    "area_natureza",
    "area_matematica",
]

_CONECTIVOS = [
    "portanto", "logo", "assim", "pois", "porque", "entretanto",
    "contudo", "todavia", "porém", "contudo", "embora", "apesar",
    "visto que", "haja vista", "além disso", "no entanto", "mas",
    "caso", "quando", "desde que",
]
_NEGACOES = {"não", "nenhum", "nunca", "jamais", "exceto", "salvo", "nem"}


def encontrar_pdf(ano: int, dia: int, base_dir: Path) -> "Path | None":
    """Localiza o PDF P1 AMARELO do dia indicado para o ano dado."""
    provas_dir = base_dir / f"microdados_enem_{ano}" / "PROVAS E GABARITOS"
    if not provas_dir.exists():
        return None
    excluir = {"ESPANHOL", "INGLES", "AMPLIADA", "SUPER", "LIBRAS", "LEDOR"}
    # *CAD* garante que pegamos o caderno de questões, não o gabarito (GAB)
    candidatos = sorted(
        p for p in provas_dir.glob(f"*P1*CAD*DIA_{dia}*AMARELO*.pdf")
        if not any(x in p.name.upper() for x in excluir)
    )
    return candidatos[0] if candidatos else None


def extrair_questoes_pdf(pdf_path: Path) -> "dict[int, str]":
    """
    Extrai o texto de cada questão de um caderno ENEM em PDF.
    Retorna {numero_questao: texto_do_enunciado}.

    Os números aparecem isolados em linha própria no layout impresso P1.
    Para questões com figura, o texto extraído será curto (proxy: tem_imagem=1).
    """
    doc = fitz.open(str(pdf_path))
    texto_completo = "".join(page.get_text() + "\n" for page in doc)
    doc.close()

    # Padrão primário: número isolado em linha (layout ENEM impresso)
    padrao = re.compile(r"(?:^|\n)\s*(\d{1,3})\s*\n", re.MULTILINE)
    matches = list(padrao.finditer(texto_completo))

    # Fallback: "QUESTÃO N" explícito (alguns anos digitais)
    if len(matches) < 10:
        padrao = re.compile(r"(?:QUESTÃO|Questão)\s+(\d{1,3})", re.MULTILINE)
        matches = list(padrao.finditer(texto_completo))

    questoes: dict[int, str] = {}
    for i, m in enumerate(matches):
        num = int(m.group(1))
        if num < 1 or num > 200:
            continue
        inicio = m.end()
        fim = matches[i + 1].start() if i + 1 < len(matches) else len(texto_completo)
        questoes[num] = texto_completo[inicio:fim].strip()

    return questoes


def extrair_features(texto: str, area: str = "") -> dict:
    """
    Computa features leves de texto para uma questão.
    Inclui one-hot de área quando informada.
    """
    if not isinstance(texto, str) or not texto.strip():
        return _features_vazias(area)

    texto = re.sub(r"\n+", " ", texto)
    texto = re.sub(r" +", " ", texto).strip()

    palavras = texto.split()
    sentencas = [s.strip() for s in re.split(r"[.!?]", texto) if len(s.strip()) > 3]

    n_palavras = len(palavras)
    n_sentencas = max(len(sentencas), 1)
    media_sent = n_palavras / n_sentencas
    pct_longas = sum(1 for p in palavras if len(p) > 8) / max(n_palavras, 1)

    texto_lower = texto.lower()
    palavras_lower = set(texto_lower.split())
    n_conectivos = sum(1 for c in _CONECTIVOS if c in texto_lower)
    tem_negacao = int(bool(_NEGACOES & palavras_lower))
    tem_formula = int(bool(re.search(
        r"[\d]+[.,][\d]+|[+×÷=<>²³√∑∫∂∆]|[0-9]+\s*[%°]|\bx\b|\by\b", texto
    )))
    # Questões com figura dominante têm pouco texto extraível do PDF
    tem_imagem = int(n_palavras < 30)

    feats = {
        "n_palavras": n_palavras,
        "media_palavras_por_sentenca": round(media_sent, 2),
        "pct_palavras_longas": round(pct_longas, 4),
        "n_conectivos_logicos": n_conectivos,
        "tem_negacao": tem_negacao,
        "tem_formula_ou_numero": tem_formula,
        "tem_imagem": tem_imagem,
    }
    for sg, col in AREA_NAMES.items():
        feats[col] = int(area == sg)
    return feats


def _features_vazias(area: str = "") -> dict:
    feats = {
        "n_palavras": 0,
        "media_palavras_por_sentenca": 0.0,
        "pct_palavras_longas": 0.0,
        "n_conectivos_logicos": 0,
        "tem_negacao": 0,
        "tem_formula_ou_numero": 0,
        "tem_imagem": 1,
    }
    for sg, col in AREA_NAMES.items():
        feats[col] = int(area == sg)
    return feats


def carregar_itens_ano(ano: int, base_dir: Path) -> pd.DataFrame:
    """
    Carrega e processa itens de um ano do ENEM:
    - Lê ITENS_PROVA_<ANO>.csv
    - Remove itens abandonados e B fora de [-4, 4]
    - Calcula difficulty_score via sigmóide: 1 / (1 + exp(-B))
    - Extrai texto das provas PDF P1 AMARELO por CO_POSICAO
    - Computa features de texto
    """
    csv_path = (
        base_dir / f"microdados_enem_{ano}" / "DADOS" / f"ITENS_PROVA_{ano}.csv"
    )
    if not csv_path.exists():
        print(f"  [AVISO] CSV não encontrado: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(
        csv_path,
        sep=";",
        encoding="latin-1",
        usecols=["CO_POSICAO", "SG_AREA", "CO_ITEM",
                 "IN_ITEM_ABAN", "NU_PARAM_B", "TX_COR", "CO_PROVA"],
        dtype={
            "CO_POSICAO": "Int64",
            "CO_ITEM": "Int64",
            "CO_PROVA": "Int64",
            "IN_ITEM_ABAN": "Int64",
            "NU_PARAM_B": str,
            "TX_COR": str,
            "SG_AREA": str,
        },
    )

    df = df[df["IN_ITEM_ABAN"] != 1].copy()

    # Vírgula decimal → ponto
    df["NU_PARAM_B"] = (
        df["NU_PARAM_B"].str.replace(",", ".", regex=False).astype(float)
    )
    df = df.dropna(subset=["NU_PARAM_B"])
    # Não há limite teórico em B — a sigmóide lida corretamente com qualquer valor.
    # O INEP calibrou todos os itens com IN_ITEM_ABAN=0; filtrá-los seria perda de dados.
    # (Observado: B vai de ~-1.4 a 11.14 nos dados de 2020-2024.)

    df["difficulty_score"] = (1 / (1 + np.exp(-1.0 * df["NU_PARAM_B"]))).round(6)

    # Seleciona itens AMARELA do menor CO_PROVA por área (= P1 aplicação regular)
    amarela = df[df["TX_COR"] == "AMARELA"].copy()
    if amarela.empty:
        print(f"  [AVISO] Nenhum item AMARELA em {ano}.")
        return pd.DataFrame()

    p1_map = amarela.groupby("SG_AREA")["CO_PROVA"].min().to_dict()
    amarela = amarela[
        amarela.apply(lambda r: r["CO_PROVA"] == p1_map.get(r["SG_AREA"]), axis=1)
    ].copy()

    # Extrai texto dos PDFs
    questoes: dict[int, str] = {}
    for dia, areas in [(1, DAY1_AREAS), (2, DAY2_AREAS)]:
        pdf = encontrar_pdf(ano, dia, base_dir)
        if pdf:
            print(f"  PDF DIA {dia}: {pdf.name}")
            questoes.update(extrair_questoes_pdf(pdf))
        else:
            print(f"  [AVISO] PDF DIA {dia} não encontrado para {ano}.")

    amarela["texto"] = amarela["CO_POSICAO"].map(lambda pos: questoes.get(int(pos), ""))

    # reset_index antes do apply para que feats_df herde índice 0-based e o
    # pd.concat não crie linhas fantasmas por desalinhamento de índice
    amarela = amarela.reset_index(drop=True)
    feats_df = amarela.apply(
        lambda r: extrair_features(r["texto"], r["SG_AREA"]), axis=1
    ).apply(pd.Series)
    amarela = pd.concat([amarela, feats_df], axis=1)

    amarela["ano"] = ano
    amarela = amarela.rename(columns={"SG_AREA": "area", "CO_ITEM": "co_item"})

    colunas = ["ano", "co_item", "area", "texto", "difficulty_score"] + FEATURES
    return amarela[[c for c in colunas if c in amarela.columns]]


def gerar_eda(df: pd.DataFrame, output_dir: Path) -> None:
    """Gera histograma por área e boxplot por ano do difficulty_score."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for area in sorted(df["area"].dropna().unique()):
        subset = df[df["area"] == area]["difficulty_score"]
        axes[0].hist(subset, bins=30, alpha=0.55, label=area)
    axes[0].set_title("Distribuição do difficulty_score por Área")
    axes[0].set_xlabel("difficulty_score")
    axes[0].set_ylabel("Frequência")
    axes[0].legend()

    anos = sorted(df["ano"].unique())
    axes[1].boxplot(
        [df[df["ano"] == a]["difficulty_score"].values for a in anos],
        tick_labels=anos,
    )
    axes[1].set_title("difficulty_score por Ano")
    axes[1].set_xlabel("Ano")
    axes[1].set_ylabel("difficulty_score")

    plt.tight_layout()
    out = output_dir / "eda_difficulty_score.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  EDA salva em {out}")


def build_dataset(
    anos: "list[int]" = ANOS,
    base_dir: Path = BASE_DIR,
) -> pd.DataFrame:
    """
    Constrói o dataset completo iterando sobre os anos indicados.
    Salva em data/processed/dataset_enem_dificuldade.csv e gera EDA.
    """
    dfs = []
    for ano in anos:
        print(f"\nProcessando {ano}...")
        df_ano = carregar_itens_ano(ano, base_dir)
        if not df_ano.empty:
            dfs.append(df_ano)
            print(f"  {len(df_ano)} itens adicionados.")

    if not dfs:
        raise ValueError("Nenhum dado carregado. Verifique os caminhos.")

    dataset = pd.concat(dfs, ignore_index=True)

    out = PROCESSED_DIR / "dataset_enem_dificuldade.csv"
    dataset.to_csv(out, index=False, encoding="utf-8")
    print(f"\nDataset salvo: {out}  ({len(dataset)} linhas)")
    print("\nDistribuição do difficulty_score:")
    print(dataset["difficulty_score"].describe().round(4))
    print("\nContagem por área:")
    print(dataset["area"].value_counts())

    gerar_eda(dataset, PROCESSED_DIR)
    return dataset


if __name__ == "__main__":
    build_dataset()
