#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_maritaca_irt.py — Unifica a base ENEM da Maritaca (texto limpo + gabarito) com
a dificuldade IRT (NU_PARAM_B) dos microdados INEP, via join POSICIONAL por cor.

Gera (no disco D, ao lado deste script):
  - raw/2022.jsonl, raw/2023.jsonl, raw/2024.jsonl   (Maritaca bruta, p/ reprodutibilidade)
  - maritaca_enem_irt.csv                             (dataset unificado, 540 linhas)
  - prompts_calibracao.txt                            (easy:/hard: p/ server_smoke_test.py)

NÃO altera o app nem o dataset IRT original (data/processed/dataset_enem_dificuldade.csv).

--------------------------------------------------------------------------------
LÓGICA DA DETECÇÃO DE COR (o ponto não-óbvio do join)
--------------------------------------------------------------------------------
A dificuldade IRT (NU_PARAM_B) é calibrada por item (CO_ITEM) e INDEPENDE da cor do
caderno. Já a POSIÇÃO de cada item (CO_POSICAO, 1..180) muda conforme a cor, porque o
ENEM embaralha a ordem das questões por cor. A base Maritaca numera as questões
"questao_01..questao_180" seguindo UMA cor específica — que não é a AMARELA usada pelo
pipeline IRT original.

Para descobrir qual cor a Maritaca usou, comparamos o GABARITO (coluna TX_GABARITO do
ITENS_PROVA, letra A–E), que é confiável, com o `label` da Maritaca em cada posição,
para cada cor com 180 posições. A cor com maior concordância é a fonte da Maritaca.
(Empiricamente: VERDE em 2022/2023 e ROXA em 2024, ~95–98%.) Um join posicional ingênuo
contra a AMARELA acertaria só ~20%.

Feita a cor: Maritaca.posição -> (CO_ITEM, gabarito) da cor correta -> difficulty_score
= sigmoide(NU_PARAM_B) daquele CO_ITEM. Assim aproveitamos o TEXTO LIMPO da Maritaca
(a coluna `texto` do CSV IRT é ruidosa/desalinhada) + o GABARITO + a DIFICULDADE.

Itens sem IRT (anulados / sem NU_PARAM_B, ~9/540): mantidos no CSV com difficulty_score
VAZIO e has_irt=False (documentado), para não perder a cobertura de texto/gabarito.
"""

import csv
import io
import json
import math
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

# ── Caminhos (tudo relativo à raiz do repo classificador-questões) ────────────
HERE = Path(__file__).resolve().parent          # .../data/maritaca
REPO = HERE.parents[1]                           # .../classificador-questões
RAW_DIR = HERE / "raw"
IRT_ITENS = lambda y: REPO / f"microdados_enem_{y}" / "DADOS" / f"ITENS_PROVA_{y}.csv"
OUT_CSV = HERE / "maritaca_enem_irt.csv"
OUT_PROMPTS = HERE / "prompts_calibracao.txt"

ANOS = [2022, 2023, 2024]
HF_URL = "https://huggingface.co/datasets/maritaca-ai/enem/resolve/main/{ano}.jsonl"

N_CALIB = 15  # ~15 easy + 15 hard


# ── 1. Obter a Maritaca (baixa do HuggingFace se ainda não houver local) ──────
def baixar_maritaca():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for ano in ANOS:
        dest = RAW_DIR / f"{ano}.jsonl"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = HF_URL.format(ano=ano)
        print(f"  baixando {url}")
        with urllib.request.urlopen(url, timeout=120) as resp:
            dest.write_bytes(resp.read())


def carregar_maritaca():
    """Retorna {ano: {posicao: registro}}."""
    by_year = defaultdict(dict)
    for ano in ANOS:
        for line in (RAW_DIR / f"{ano}.jsonl").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pos = int(re.search(r"(\d+)", r["id"]).group(1))
            by_year[ano][pos] = r
    return by_year


# ── 2. Ler ITENS_PROVA (todas as cores) ───────────────────────────────────────
def carregar_itens(ano):
    with open(IRT_ITENS(ano), encoding="latin-1") as f:
        return list(csv.DictReader(f, delimiter=";"))


def parse_b(row):
    b = (row.get("NU_PARAM_B") or "").replace(",", ".")
    try:
        return float(b)
    except ValueError:
        return None


def difficulty_por_item(itens):
    """difficulty_score = sigmoide(NU_PARAM_B), por CO_ITEM não-anulado."""
    d = {}
    for r in itens:
        b = parse_b(r)
        aban = str(r.get("IN_ITEM_ABAN", "")).strip() in ("1", "1.0")
        if b is not None and not aban:
            d[str(int(float(r["CO_ITEM"])))] = round(1 / (1 + math.exp(-b)), 6)
    return d


# ── 3. Detectar a cor da Maritaca por concordância de gabarito ────────────────
def mapa_posicao_por_cor(itens):
    """{cor: {posicao: (co_item, gabarito)}} — apenas cores com >=180 posições."""
    cor_pos = defaultdict(dict)
    for r in itens:
        try:
            p = int(float(r["CO_POSICAO"]))
        except (TypeError, ValueError):
            continue
        gab = (r.get("TX_GABARITO") or "").strip().upper()
        cor_pos[r["TX_COR"]][p] = (str(int(float(r["CO_ITEM"]))), gab)
    return {c: d for c, d in cor_pos.items() if len(d) >= 180}


def detectar_cor(cor_pos_full, maritaca_ano):
    melhor, melhor_taxa, melhor_map = None, -1.0, None
    for cor, pos2 in cor_pos_full.items():
        acertos = total = 0
        for p, m in maritaca_ano.items():
            lab = str(m.get("label", "")).strip().upper()
            if p in pos2 and pos2[p][1] in "ABCDE" and lab in "ABCDE":
                total += 1
                acertos += pos2[p][1] == lab
        taxa = acertos / total if total else 0.0
        if taxa > melhor_taxa:
            melhor, melhor_taxa, melhor_map = cor, taxa, pos2
    return melhor, melhor_taxa, melhor_map


def area_da_posicao(p):
    return "LC" if p <= 45 else "CH" if p <= 90 else "CN" if p <= 135 else "MT"


# ── Limpeza leve do enunciado (para os prompts) ───────────────────────────────
def limpar_texto(s):
    s = s or ""
    s = s.replace(" ", " ").replace("​", "")
    s = re.sub(r"#{1,6}\s*", " ", s)      # markdown headings
    s = re.sub(r"\[\.\.\.\]", " ", s)     # cortes "[...]"
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ── 4. Construir o dataset unificado ──────────────────────────────────────────
def build():
    baixar_maritaca()
    maritaca = carregar_maritaca()

    rows = []
    resumo = {}
    for ano in ANOS:
        itens = carregar_itens(ano)
        diff = difficulty_por_item(itens)
        cor_full = mapa_posicao_por_cor(itens)
        cor, taxa, pos2 = detectar_cor(cor_full, maritaca[ano])
        resumo[ano] = (cor, taxa)
        print(f"  {ano}: cor detectada = {cor} (gabarito {taxa:.1%})")

        for pos in sorted(maritaca[ano]):
            m = maritaca[ano][pos]
            co_item, _gab = pos2.get(pos, (None, None))
            dscore = diff.get(co_item, "") if co_item else ""
            rows.append({
                "id": m["id"],
                "ano": ano,
                "area": area_da_posicao(pos),
                "posicao": pos,
                "co_item": co_item or "",
                "question": limpar_texto(m.get("question", "")),
                "alternatives": json.dumps(m.get("alternatives", []), ensure_ascii=False),
                "label": m.get("label", ""),
                "IU": bool(m.get("IU")),
                "description": " | ".join(m.get("description") or []),
                "difficulty_score": dscore,
                "has_irt": dscore != "",
                "caderno_cor": cor,
            })

    # CSV
    cols = ["id", "ano", "area", "posicao", "co_item", "question", "alternatives",
            "label", "IU", "description", "difficulty_score", "has_irt", "caderno_cor"]
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)

    com_irt = [r for r in rows if r["has_irt"]]
    print(f"\n  {OUT_CSV.name}: {len(rows)} linhas | com IRT: {len(com_irt)} | sem IRT: {len(rows) - len(com_irt)}")

    # ── 5. prompts_calibracao.txt (IU=false, extremos de dificuldade) ─────────
    puros = [r for r in com_irt if not r["IU"] and r["question"]]
    puros.sort(key=lambda r: float(r["difficulty_score"]))
    easy = puros[:N_CALIB]
    hard = puros[-N_CALIB:]
    with open(OUT_PROMPTS, "w", encoding="utf-8") as f:
        f.write("# Conjunto de calibração de thresholds (ENEM/Maritaca + IRT)\n")
        f.write("# easy = decil inferior de difficulty_score (IRT); hard = decil superior.\n")
        f.write("# Fonte: maritaca_enem_irt.csv (IU=false, texto puro).\n")
        for r in easy:
            f.write(f"easy: {r['question']}\n")
        for r in hard:
            f.write(f"hard: {r['question']}\n")
    print(f"  {OUT_PROMPTS.name}: {len(easy)} easy + {len(hard)} hard "
          f"(de {len(puros)} itens texto-puro com IRT)")

    print("\n  Cores detectadas:", {a: c for a, (c, _t) in resumo.items()})
    return rows, easy, hard


if __name__ == "__main__":
    print("Construindo dataset unificado Maritaca + IRT...")
    build()
    print("OK")
