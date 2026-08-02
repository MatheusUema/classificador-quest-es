#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recompute_debias.py — Recalcula APENAS a SEÇÃO 4 (acurácia debiased / consistência)
do experimento de debiasing de opção, LENDO um CSV de detalhe já existente
(resultados_debias_<model>.csv). NÃO consulta o servidor — é offline e barato.

Motivo: no CSV gerado pela versão antiga do evaluate_option_debias.py a seção 4 foi
agrupada por `id`, que REPETE entre os anos 2022/2023/2024 na base Maritaca — juntando
3 anos × 5 permutações = 15 linhas na mesma "questão" (histograma indo até 15, n=175).
Aqui reconstruímos a questão corretamente e recalculamos.

Agrupamento (preferência, com validação de tamanho <= 5):
  1) coluna `qid` (id + ano), se presente;
  2) senão, `id` + `ano`;
  3) senão (ou se o agrupamento acima gerar grupos > 5 linhas), BLOCOS CONSECUTIVOS
     de 5 linhas — como o experimento processa cada questão em 5 permutações
     consecutivas (perm_idx 0..4), isso reconstrói as questões.

Recalcula: acurácia de ordem única (perm_idx 0), acurácia debiased (voto majoritário
do conteúdo escolhido nas 5), consistência (mesmo conteúdo em >= k das 5) e o
histograma de concordância (agora <= 5). Só stdlib.

Uso:
  python recompute_debias.py --in resultados_debias_qwen2.5-1.5b.csv
  python recompute_debias.py --in resultados_debias_qwen2.5-1.5b.csv --kmin 3
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LETTERS = "ABCDE"


def parse_bool(v):
    s = str(v).strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    return None


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_groups(rows):
    """Retorna (metodo, [(chave, [linhas])]). Valida tamanho <= 5; senão cai em blocos."""
    for method in ("qid", "id_ano"):
        if method == "qid" and not (rows and rows[0].get("qid")):
            continue
        if method == "id_ano" and not (rows and rows[0].get("ano")):
            continue
        groups, order, ok = {}, [], True
        for r in rows:
            if method == "qid":
                k = (r.get("qid") or "").strip()
            else:
                k = f"{(r.get('id') or '').strip()}_{(r.get('ano') or '').strip()}"
            if not k or k.endswith("_"):        # ano vazio -> chave ambígua
                ok = False
                break
            if k not in groups:
                order.append(k)
                groups[k] = []
            groups[k].append(r)
        if ok and groups and max(len(v) for v in groups.values()) <= 5:
            return method, [(k, groups[k]) for k in order]
    # fallback robusto: blocos consecutivos de 5 (perm_idx 0..4)
    return "blocos_de_5", [(f"bloco_{i // 5}", rows[i:i + 5])
                           for i in range(0, len(rows), 5)]


def section4(glist, kmin):
    n_deb = correct_deb = 0
    n_single = correct_single = 0
    hist = Counter()
    consist_ge = 0
    sizes = Counter()
    for _key, recs in glist:
        sizes[len(recs)] += 1
        label = (recs[0].get("label") or "").strip().upper()
        c = LETTERS.find(label)

        # ordem única = perm_idx 0 (ou 1ª linha do bloco se não houver perm_idx)
        p0 = next((x for x in recs if str(x.get("perm_idx", "")).strip() == "0"), None)
        if p0 is None and recs:
            p0 = recs[0]
        a0 = parse_bool(p0.get("acertou")) if p0 else None
        if a0 is not None:
            n_single += 1
            correct_single += 1 if a0 else 0

        contents = []
        for x in recs:
            v = str(x.get("conteudo_escolhido_id", "")).strip()
            if v != "":
                try:
                    contents.append(int(v))
                except ValueError:
                    pass
        if not contents:
            continue
        top_content, top_n = Counter(contents).most_common(1)[0]
        hist[top_n] += 1
        if top_n >= kmin:
            consist_ge += 1
        n_deb += 1
        if c >= 0 and top_content == c:
            correct_deb += 1

    return {
        "n_grupos": len(glist),
        "sizes": dict(sizes),
        "acc_single": (correct_single / n_single) if n_single else float("nan"),
        "n_single": n_single,
        "acc_debiased": (correct_deb / n_deb) if n_deb else float("nan"),
        "n_debiased": n_deb,
        "consist_hist": dict(sorted(hist.items())),
        "frac_consist_ge": (consist_ge / n_deb) if n_deb else float("nan"),
        "kmin": kmin,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Recalcula a secao 4 (debiased/consistencia) de um CSV de debiasing.")
    ap.add_argument("--in", dest="inp",
                    default=str(HERE / "resultados_debias_qwen2.5-1.5b.csv"),
                    help="CSV de detalhe (resultados_debias_<model>.csv)")
    ap.add_argument("--kmin", type=int, default=3,
                    help="k da consistencia (mesmo conteudo em >= k das 5). Default 3")
    args = ap.parse_args()

    path = Path(args.inp)
    if not path.exists():
        print(f"Arquivo nao encontrado: {path}")
        return 1
    rows = read_rows(path)
    if not rows:
        print("CSV vazio.")
        return 1

    method, glist = make_groups(rows)
    S = section4(glist, args.kmin)

    print("=" * 60)
    print("SECAO 4 RECALCULADA (debiased / consistencia)")
    print("=" * 60)
    print(f"CSV: {path.name} | linhas: {len(rows)}")
    print(f"Agrupamento: {method} | questoes reconstruidas: {S['n_grupos']}")
    print(f"Tamanhos de grupo (deveria ser 5): {S['sizes']}")
    if S["n_grupos"] and max(S["sizes"]) > 5:
        print("  ! ATENCAO: ha grupo com > 5 linhas — agrupamento suspeito.")
    print("-" * 60)
    print(f"Acuracia ordem unica (perm 0): {S['acc_single']*100:.1f}%  (n={S['n_single']})")
    print(f"Acuracia debiased (voto 5):    {S['acc_debiased']*100:.1f}%  (n={S['n_debiased']})")
    print(f"Consistencia (mesmo conteudo em >= {args.kmin} das 5): "
          f"{S['frac_consist_ge']*100:.1f}% das questoes")
    print(f"Histograma de concordancia (max iguais das 5): {S['consist_hist']}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
