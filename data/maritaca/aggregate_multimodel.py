#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_multimodel.py — Junta os CSVs de vários modelos locais (gerados por
evaluate_local_accuracy.py com --model-name) e produz um COMPARATIVO entre modelos.

NÃO consulta o servidor. Só lê CSVs já existentes.

--------------------------------------------------------------------------------
FLUXO DE USO (multi-modelo, um modelo de cada vez, mesma porta 8080)
  Para CADA modelo local (Gemma-3-1B, Qwen2.5-0.5B, Qwen2.5-1.5B, Llama-3.2-1B):
    (a) suba o llama-server apontando pro GGUF daquele modelo na porta 8080:
          llama-server -m <modelo>.gguf --port 8080 --n-probs 5
    (b) com o servidor no ar, rode a avaliação com um rótulo:
          python evaluate_local_accuracy.py --url http://127.0.0.1:8080 --model-name gemma-3-1b
          python evaluate_local_accuracy.py --url http://127.0.0.1:8080 --model-name qwen2.5-0.5b
          python evaluate_local_accuracy.py --url http://127.0.0.1:8080 --model-name qwen2.5-1.5b
          python evaluate_local_accuracy.py --url http://127.0.0.1:8080 --model-name llama-3.2-1b
        (cada um gera resultados_acerto_<rótulo>.csv)
    (c) DEPOIS de rodar os 4, rode UMA vez este agregador:
          python aggregate_multimodel.py
--------------------------------------------------------------------------------
ENTRADAS
  - Por padrão: glob resultados_acerto_*.csv na pasta deste script.
  - Ou explicitamente: python aggregate_multimodel.py a.csv b.csv c.csv
  O rótulo de cada modelo vem da coluna `model` (gravada pelo evaluate); se estiver
  vazia (CSV antigo), usa-se o nome do arquivo (stem sem o prefixo resultados_acerto_).

MÉTRICAS por modelo (mesma definição do analyze_accuracy.py, aqui em stdlib):
  acurácia global e por área (LC/CH/CN/MT), confiança média em acerto/erro,
  ROC AUC global e por área, e ECE (calibração).

SAÍDAS
  - comparativo_modelos.csv  (uma linha por modelo × métrica)  -> pasta do script.
  - Tabela lado a lado no console (modelos nas colunas, métricas nas linhas),
    destacando acurácia global, AUC global e ECE.
  - Se matplotlib estiver disponível: fig_cmp_acc_area.png (barras de acurácia por
    área agrupadas por modelo) e fig_cmp_auc_global.png (AUC global por modelo),
    salvas em ./analise/.

Só stdlib para o núcleo; matplotlib é OPCIONAL (só para os PNGs). Reusa as funções
de métrica de analyze_accuracy.py quando importável; senão, usa cópias locais.
"""

import argparse
import csv
import glob
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AREAS = ["LC", "CH", "CN", "MT"]
DEFAULT_GLOB = "resultados_acerto_*.csv"
OUT_CSV = HERE / "comparativo_modelos.csv"
FIG_DIR = HERE / "analise"


# ── Métricas: reuso de analyze_accuracy.py se possível, senão cópias stdlib ────
def _local_auc(scores, labels):
    """Mann-Whitney (ranks médios) = P(conf_acerto > conf_erro). 0/1 em labels."""
    n = len(scores)
    if n == 0:
        return None
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    P = sum(labels)
    N = n - P
    if P == 0 or N == 0:
        return None
    sp = sum(ranks[i] for i in range(n) if labels[i] == 1)
    return (sp - P * (P + 1) / 2.0) / (P * N)


try:  # reuso opcional (analyze_accuracy importa numpy/matplotlib no topo)
    from analyze_accuracy import auc as _auc
except Exception:  # ImportError ou deps ausentes -> usa cópia local (stdlib)
    _auc = _local_auc


def ece_score(confs, corrects, nbins=10):
    """Expected Calibration Error (stdlib), bins uniformes em [0,1]."""
    n = len(confs)
    if n == 0:
        return float("nan")
    ece = 0.0
    for b in range(nbins):
        lo = b / nbins
        hi = (b + 1) / nbins
        idx = [i for i in range(n)
               if (confs[i] >= lo and (confs[i] < hi if b < nbins - 1 else confs[i] <= hi))]
        if not idx:
            continue
        acc_b = sum(corrects[i] for i in idx) / len(idx)
        conf_b = sum(confs[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(acc_b - conf_b)
    return ece


# ── Leitura ───────────────────────────────────────────────────────────────────
def label_from_file(path):
    stem = Path(path).stem
    for pre in ("resultados_acerto_", "resultados_"):
        if stem.startswith(pre):
            return stem[len(pre):]
    return stem


def load_file(path):
    """Retorna (label, rows). label vem da coluna `model` (se preenchida) ou do arquivo."""
    rows = []
    model_label = ""
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ac = str(r.get("acertou", "")).strip().lower()
            if ac not in ("true", "false"):
                continue
            try:
                conf = float(r.get("confianca", ""))
            except (TypeError, ValueError):
                continue
            if not model_label:
                model_label = str(r.get("model", "")).strip()
            rows.append({
                "area": str(r.get("area", "")).upper(),
                "correct": 1 if ac == "true" else 0,
                "conf": conf,
            })
    if not model_label:
        model_label = label_from_file(path)
    return model_label, rows


# ── Cálculo por modelo ────────────────────────────────────────────────────────
def compute_metrics(rows):
    m = {"n": len(rows)}
    if not rows:
        return m
    m["acc_global"] = sum(r["correct"] for r in rows) / len(rows)
    corr = [r["conf"] for r in rows if r["correct"] == 1]
    err = [r["conf"] for r in rows if r["correct"] == 0]
    m["conf_acerto"] = (sum(corr) / len(corr)) if corr else float("nan")
    m["conf_erro"] = (sum(err) / len(err)) if err else float("nan")
    m["auc_global"] = _auc([r["conf"] for r in rows], [r["correct"] for r in rows])
    m["ece"] = ece_score([r["conf"] for r in rows], [r["correct"] for r in rows])
    for a in AREAS:
        sub = [r for r in rows if r["area"] == a]
        if sub:
            m[f"acc_{a}"] = sum(r["correct"] for r in sub) / len(sub)
            m[f"auc_{a}"] = _auc([r["conf"] for r in sub], [r["correct"] for r in sub])
            m[f"n_{a}"] = len(sub)
        else:
            m[f"acc_{a}"] = None
            m[f"auc_{a}"] = None
            m[f"n_{a}"] = 0
    return m


# ── Saída: CSV longo (modelo × métrica) ───────────────────────────────────────
METRIC_ORDER = (
    ["n", "acc_global", "conf_acerto", "conf_erro", "auc_global", "ece"]
    + [f"acc_{a}" for a in AREAS]
    + [f"auc_{a}" for a in AREAS]
    + [f"n_{a}" for a in AREAS]
)


def write_comparativo_csv(models, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "metric", "value"])
        for label in models:
            m = models[label]
            for k in METRIC_ORDER:
                v = m.get(k)
                if v is None:
                    sval = ""
                elif isinstance(v, float):
                    sval = "" if math.isnan(v) else f"{v:.6f}"
                else:
                    sval = str(v)
                wr.writerow([label, k, sval])


# ── Saída: tabela no console (modelos nas colunas) ────────────────────────────
def fmt(v, kind):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "f3":
        return f"{v:.3f}"
    return str(v)


ROWS_SPEC = [
    ("»» Acuracia GLOBAL", "acc_global", "pct"),
    ("   Acuracia LC", "acc_LC", "pct"),
    ("   Acuracia CH", "acc_CH", "pct"),
    ("   Acuracia CN", "acc_CN", "pct"),
    ("   Acuracia MT", "acc_MT", "pct"),
    ("   Conf. media ACERTO", "conf_acerto", "f3"),
    ("   Conf. media ERRO", "conf_erro", "f3"),
    ("»» AUC GLOBAL", "auc_global", "f3"),
    ("   AUC LC", "auc_LC", "f3"),
    ("   AUC CH", "auc_CH", "f3"),
    ("   AUC CN", "auc_CN", "f3"),
    ("   AUC MT", "auc_MT", "f3"),
    ("»» ECE (calibracao)", "ece", "f3"),
    ("   N avaliadas", "n", "int"),
]


def print_table(models):
    labels = list(models.keys())
    col_w = max(12, max((len(l) for l in labels), default=12))
    label_w = max(len(r[0]) for r in ROWS_SPEC)
    header = " " * label_w + " | " + " | ".join(l.rjust(col_w) for l in labels)
    print("=" * len(header))
    print("COMPARATIVO MULTI-MODELO  (»» = metricas-chave)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, key, kind in ROWS_SPEC:
        cells = " | ".join(fmt(models[l].get(key), kind).rjust(col_w) for l in labels)
        print(name.ljust(label_w) + " | " + cells)
    print("=" * len(header))


# ── Figuras opcionais (matplotlib) ────────────────────────────────────────────
def make_figures(models):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib indisponivel — PNGs comparativos nao gerados.)")
        return []
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = list(models.keys())
    colors = ["#2e86ab", "#e8a13a", "#5aa9c9", "#d1495b", "#7a7a7a", "#6a4c93"]
    saved = []

    # 1) barras de acuracia por area, agrupadas por modelo
    fig1 = FIG_DIR / "fig_cmp_acc_area.png"
    x = range(len(AREAS))
    nmod = len(labels)
    bw = 0.8 / max(nmod, 1)
    plt.figure(figsize=(8, 5))
    for j, lab in enumerate(labels):
        vals = [(models[lab].get(f"acc_{a}") or 0) * 100 for a in AREAS]
        offs = [i - 0.4 + bw * (j + 0.5) for i in x]
        plt.bar(offs, vals, width=bw, label=lab, color=colors[j % len(colors)])
    plt.axhline(20, color="gray", ls="--", lw=1, label="acaso (20%)")
    plt.xticks(list(x), AREAS)
    plt.ylabel("acuracia (%)"); plt.ylim(0, 100)
    plt.title("Acuracia por area — comparativo entre modelos")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(fig1, dpi=140); plt.close()
    saved.append(fig1)

    # 2) AUC global por modelo
    fig2 = FIG_DIR / "fig_cmp_auc_global.png"
    aucs = [models[l].get("auc_global") for l in labels]
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(range(len(labels)), [a if a is not None else 0 for a in aucs],
                   color=[colors[j % len(colors)] for j in range(len(labels))])
    plt.axhline(0.5, color="gray", ls="--", lw=1, label="aleatorio (0.5)")
    plt.axhline(0.7, color="green", ls=":", lw=1, label="sinal util (0.7)")
    for b, a in zip(bars, aucs):
        if a is not None:
            plt.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.3f}",
                     ha="center", fontsize=9)
    plt.xticks(range(len(labels)), labels, rotation=15, ha="right")
    plt.ylabel("ROC AUC global"); plt.ylim(0, 1)
    plt.title("Qualidade do sinal de confianca (AUC global) por modelo")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(fig2, dpi=140); plt.close()
    saved.append(fig2)
    return saved


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Agrega resultados_acerto_*.csv de varios modelos num comparativo.")
    ap.add_argument("inputs", nargs="*",
                    help="CSVs de resultado (opcional). Default: glob resultados_acerto_*.csv")
    ap.add_argument("--glob", default=DEFAULT_GLOB,
                    help=f"padrao de busca quando nao ha inputs. Default {DEFAULT_GLOB}")
    ap.add_argument("--out", default=str(OUT_CSV),
                    help="CSV comparativo de saida (modelo x metrica)")
    ap.add_argument("--no-fig", action="store_true", help="nao gerar PNGs")
    args = ap.parse_args()

    if args.inputs:
        files = list(args.inputs)
    else:
        files = sorted(glob.glob(str(HERE / args.glob)))
    if not files:
        print(f"Nenhum CSV encontrado (glob: {args.glob}).")
        return 1

    models = {}
    print(f"Arquivos lidos: {len(files)}")
    for path in files:
        label, rows = load_file(path)
        if not rows:
            print(f"  - {Path(path).name}: sem linhas validas, ignorado.")
            continue
        if label in models:
            print(f"  ! rotulo repetido '{label}' — usando o ultimo arquivo ({Path(path).name}).")
        models[label] = compute_metrics(rows)
        print(f"  - {Path(path).name}: modelo='{label}', n={len(rows)}")
    if not models:
        print("Nenhum modelo com dados validos.")
        return 1

    write_comparativo_csv(models, Path(args.out))
    print(f"\nComparativo (modelo x metrica): {args.out}\n")
    print_table(models)
    if not args.no_fig:
        saved = make_figures(models)
        for p in saved:
            print("Figura:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
