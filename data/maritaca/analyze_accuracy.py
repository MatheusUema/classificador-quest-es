#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_accuracy.py — Analisa resultados_acerto_local.csv (offline; NAO consulta o
servidor) e produz metricas por area, curva de calibracao (ECE), graficos PNG,
SIMULACAO DE POLITICAS DE ROTEAMENTO e um relatorio Markdown.

Saidas (em <out_dir>, default: ./analise):
  metrics_por_area.csv, calibracao.csv, politicas.csv
  fig_conf_dist.png, fig_roc.png, fig_acc_area.png, fig_calibracao.png, fig_policy_tradeoff.png
  relatorio.md

Uso:
  python analyze_accuracy.py                # le ./resultados_acerto_local.csv
  python analyze_accuracy.py --csv <path> --out <dir> --prob-esc 1.0,0.85 --budgets 20,40,60,80

Politica de escalonamento (proxy): "escalar" (para servidor/cloud) equivale a ACERTAR a
questao com probabilidade --prob-esc (default 1.0; roda tambem 0.85). Explicito e
parametrizavel: o tier superior e assumido forte. Isso mede o teto do ganho de escalar.
"""

import argparse
import csv
import math
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AREAS = ["LC", "CH", "CN", "MT"]


# ── Leitura robusta ──────────────────────────────────────────────────────────
def load(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ac = str(r.get("acertou", "")).strip().lower()
            if ac not in ("true", "false"):
                continue                      # erros de rede / sem avaliacao
            try:
                conf = float(r.get("confianca", ""))
            except (TypeError, ValueError):
                continue                      # sem confianca -> fora das metricas
            try:
                diff = float(r.get("difficulty_score", ""))
            except (TypeError, ValueError):
                diff = None
            rows.append({
                "id": r.get("id", ""), "area": r.get("area", "").upper(),
                "correct": 1 if ac == "true" else 0, "conf": conf, "diff": diff,
            })
    return rows


# ── Metricas ─────────────────────────────────────────────────────────────────
def auc(scores, labels):
    """Mann-Whitney (ranks medios) = P(conf_acerto > conf_erro)."""
    n = len(scores)
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
    P = sum(labels); N = n - P
    if P == 0 or N == 0:
        return None
    sp = sum(ranks[i] for i in range(n) if labels[i] == 1)
    return (sp - P * (P + 1) / 2.0) / (P * N)


def roc_curve(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    P = sum(labels); N = len(labels) - P
    tp = fp = 0
    fpr, tpr = [0.0], [0.0]
    for _, l in pairs:
        if l == 1:
            tp += 1
        else:
            fp += 1
        fpr.append(fp / N if N else 0.0)
        tpr.append(tp / P if P else 0.0)
    return fpr, tpr


def area_metrics(rows):
    out = {}
    for a in AREAS + ["GLOBAL"]:
        sub = rows if a == "GLOBAL" else [r for r in rows if r["area"] == a]
        if not sub:
            continue
        corr = [r["conf"] for r in sub if r["correct"] == 1]
        err = [r["conf"] for r in sub if r["correct"] == 0]
        out[a] = {
            "n": len(sub),
            "acc": sum(r["correct"] for r in sub) / len(sub),
            "conf_acerto": (sum(corr) / len(corr)) if corr else float("nan"),
            "conf_erro": (sum(err) / len(err)) if err else float("nan"),
            "auc": auc([r["conf"] for r in sub], [r["correct"] for r in sub]),
        }
    return out


def calibration(rows, nbins=10):
    edges = np.linspace(0, 1, nbins + 1)
    conf = np.array([r["conf"] for r in rows])
    corr = np.array([r["correct"] for r in rows], dtype=float)
    bins = []
    ece = 0.0
    N = len(rows)
    for b in range(nbins):
        lo, hi = edges[b], edges[b + 1]
        m = (conf >= lo) & (conf < hi if b < nbins - 1 else conf <= hi)
        if m.sum() == 0:
            bins.append((lo, hi, 0, float("nan"), float("nan")))
            continue
        acc_b = corr[m].mean()
        conf_b = conf[m].mean()
        ece += (m.sum() / N) * abs(acc_b - conf_b)
        bins.append((lo, hi, int(m.sum()), conf_b, acc_b))
    return bins, ece


# ── Simulacao de politicas ───────────────────────────────────────────────────
def sim_threshold(rows, p_esc, t):
    """C: escala (conf < t). Retorna (esc_rate, acuracia_final)."""
    N = len(rows)
    correct_local = sum(r["correct"] for r in rows if r["conf"] >= t)
    n_esc = sum(1 for r in rows if r["conf"] < t)
    acc = (correct_local + p_esc * n_esc) / N
    return n_esc / N, acc


def sim_area_aware(rows, p_esc, t, always=("CN", "MT")):
    """D: escala sempre CN/MT; em LC/CH escala se conf < t."""
    N = len(rows)
    n_esc = correct_local = 0
    for r in rows:
        if r["area"] in always or r["conf"] < t:
            n_esc += 1
        else:
            correct_local += r["correct"]
    acc = (correct_local + p_esc * n_esc) / N
    return n_esc / N, acc


def sim_difficulty(rows, p_esc, theta):
    """E: escala se difficulty_score >= theta (so linhas com IRT)."""
    sub = [r for r in rows if r["diff"] is not None]
    if not sub:
        return None
    N = len(sub)
    n_esc = correct_local = 0
    for r in sub:
        if r["diff"] >= theta:
            n_esc += 1
        else:
            correct_local += r["correct"]
    return n_esc / N, (correct_local + p_esc * n_esc) / N


def sweep(rows, fn, ts, p_esc):
    pts = [fn(rows, p_esc, t) for t in ts]
    pts = [p for p in pts if p is not None]
    return pts


def best_for_budget(pts, budget):
    """Maior acuracia com esc_rate <= budget."""
    ok = [(e, a) for e, a in pts if e <= budget + 1e-9]
    if not ok:
        return None
    return max(ok, key=lambda x: x[1])


# ── Graficos ─────────────────────────────────────────────────────────────────
def fig_conf_dist(rows, path):
    corr = [r["conf"] for r in rows if r["correct"] == 1]
    err = [r["conf"] for r in rows if r["correct"] == 0]
    plt.figure(figsize=(7, 4))
    bins = np.linspace(0, 1, 21)
    plt.hist(err, bins=bins, alpha=0.6, label=f"ERRO (n={len(err)})", color="#d1495b")
    plt.hist(corr, bins=bins, alpha=0.6, label=f"ACERTO (n={len(corr)})", color="#2e86ab")
    plt.axvline(np.mean(err), color="#d1495b", ls="--")
    plt.axvline(np.mean(corr), color="#2e86ab", ls="--")
    plt.xlabel("confianca"); plt.ylabel("frequencia")
    plt.title("Distribuicao de confianca: acerto vs erro")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def fig_roc(rows, ametrics, path):
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k:", lw=1, label="aleatorio")
    fpr, tpr = roc_curve([r["conf"] for r in rows], [r["correct"] for r in rows])
    plt.plot(fpr, tpr, lw=2.2, label=f"GLOBAL (AUC={ametrics['GLOBAL']['auc']:.3f})")
    for a in AREAS:
        sub = [r for r in rows if r["area"] == a]
        au = ametrics.get(a, {}).get("auc")
        if len(sub) >= 5 and au is not None:
            fpr, tpr = roc_curve([r["conf"] for r in sub], [r["correct"] for r in sub])
            plt.plot(fpr, tpr, lw=1.4, alpha=0.85, label=f"{a} (AUC={au:.3f})")
    plt.xlabel("FPR (erros aceitos)"); plt.ylabel("TPR (acertos aceitos)")
    plt.title("ROC — confianca prevendo acerto"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def fig_acc_area(ametrics, path):
    labels = [a for a in AREAS if a in ametrics]
    accs = [ametrics[a]["acc"] * 100 for a in labels]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, accs, color=["#2e86ab", "#5aa9c9", "#e8a13a", "#d1495b"])
    plt.axhline(20, color="gray", ls="--", lw=1, label="acaso (20%)")
    plt.axhline(ametrics["GLOBAL"]["acc"] * 100, color="black", ls=":", lw=1,
                label=f"global ({ametrics['GLOBAL']['acc']*100:.0f}%)")
    for b, v in zip(bars, accs):
        plt.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    plt.ylabel("acuracia local (%)"); plt.ylim(0, 100)
    plt.title("Acuracia local por area"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def fig_calibracao(bins, ece, path):
    xs = [b[3] for b in bins if b[2] > 0]
    ys = [b[4] for b in bins if b[2] > 0]
    ns = [b[2] for b in bins if b[2] > 0]
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k:", label="calibracao perfeita")
    plt.plot(xs, ys, "o-", color="#2e86ab", label="observado")
    for x, y, n in zip(xs, ys, ns):
        plt.annotate(str(n), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.xlabel("confianca media (bin)"); plt.ylabel("acuracia observada")
    plt.title(f"Curva de calibracao (ECE={ece:.3f})")
    plt.legend(); plt.xlim(0, 1); plt.ylim(0, 1)
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def fig_policy(rows, ptsC, ptsD, ptsE, p_esc, path):
    plt.figure(figsize=(7, 5))
    for pts, lab, col in [(ptsC, "C: threshold global", "#2e86ab"),
                          (ptsD, "D: area-aware (CN/MT sempre)", "#e8a13a"),
                          (ptsE, "E: pre-filtro IRT", "#7a7a7a")]:
        if not pts:
            continue
        pts = sorted(pts)
        plt.plot([e * 100 for e, a in pts], [a * 100 for e, a in pts], "-", label=lab, color=col)
    A0 = sum(r["correct"] for r in rows) / len(rows)
    plt.scatter([0], [A0 * 100], color="green", zorder=5, label=f"A: sempre local ({A0*100:.0f}%)")
    plt.scatter([100], [p_esc * 100], color="red", zorder=5,
                label=f"B: sempre escalar ({p_esc*100:.0f}%)")
    plt.xlabel("taxa de escalonamento (%) = custo"); plt.ylabel("acuracia final (%)")
    plt.title(f"Acuracia x escalonamento (p_esc={p_esc})")
    plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


# ── Relatorio ────────────────────────────────────────────────────────────────
def write_reports(outdir, rows, ametrics, bins, ece, sims, p_list, budgets, figs):
    # CSVs
    with open(outdir / "metrics_por_area.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["area", "n", "acuracia", "conf_acerto", "conf_erro", "auc"])
        for a in ["GLOBAL"] + AREAS:
            if a in ametrics:
                m = ametrics[a]
                wr.writerow([a, m["n"], f"{m['acc']:.4f}", f"{m['conf_acerto']:.4f}",
                             f"{m['conf_erro']:.4f}", "" if m["auc"] is None else f"{m['auc']:.4f}"])
    with open(outdir / "calibracao.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["bin_lo", "bin_hi", "n", "conf_media", "acuracia_obs"])
        for lo, hi, n, cf, ac in bins:
            wr.writerow([f"{lo:.2f}", f"{hi:.2f}", n,
                         "" if n == 0 else f"{cf:.4f}", "" if n == 0 else f"{ac:.4f}"])
    with open(outdir / "politicas.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["p_esc", "budget", "politica", "esc_rate", "acuracia"])
        for p in p_list:
            for bud in budgets:
                for pol in ("C", "D", "E"):
                    best = sims[p][pol + "_budget"].get(bud)
                    if best:
                        wr.writerow([p, bud, pol, f"{best[0]:.4f}", f"{best[1]:.4f}"])

    # Markdown
    A0 = ametrics["GLOBAL"]["acc"]
    md = []
    md.append("# Analise — Confianca do modelo local vs Acerto (ENEM/Maritaca)\n")
    md.append(f"Base: `resultados_acerto_local.csv` — **{len(rows)}** questoes avaliadas "
              f"(texto puro, com IRT). Acuracia local global: **{A0*100:.1f}%**.\n")
    md.append("## 1. Metricas por area\n")
    md.append("| Area | n | Acuracia | Conf. acerto | Conf. erro | AUC |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for a in ["GLOBAL"] + AREAS:
        if a in ametrics:
            m = ametrics[a]
            aucs = "n/d" if m["auc"] is None else f"{m['auc']:.3f}"
            md.append(f"| {a} | {m['n']} | {m['acc']*100:.1f}% | {m['conf_acerto']:.3f} "
                      f"| {m['conf_erro']:.3f} | {aucs} |")
    md.append("\n> CN/MT ficam proximas do acaso (20%) e a AUC nelas tende a ~0.5 — a "
              "confianca **nao** separa acerto de erro nessas areas; em LC/CH ha mais sinal.\n")
    md.append(f"![conf](./{figs['conf'].name})\n")
    md.append(f"![roc](./{figs['roc'].name})\n")
    md.append(f"![acc](./{figs['acc'].name})\n")
    md.append("## 2. Calibracao (superconfianca)\n")
    md.append(f"**ECE = {ece:.3f}.** O modelo e superconfiante: diz ~0.8+ mas acerta bem "
              "menos (a curva fica abaixo da diagonal).\n")
    md.append("| bin conf | n | conf media | acuracia obs |")
    md.append("|---|---:|---:|---:|")
    for lo, hi, n, cf, ac in bins:
        if n > 0:
            md.append(f"| {lo:.1f}-{hi:.1f} | {n} | {cf:.3f} | {ac:.3f} |")
    md.append(f"\n![calib](./{figs['calib'].name})\n")
    md.append("## 3. Simulacao de politicas de roteamento\n")
    md.append("Proxy: **escalar = acertar** com probabilidade `p_esc` (o tier superior e "
              "forte). Comparamos acuracia final x taxa de escalonamento (custo).\n")
    md.append(f"![policy](./{figs['policy'].name})\n")
    for p in p_list:
        md.append(f"\n### p_esc = {p}\n")
        md.append("| Orcamento escal. | C (threshold) | D (area-aware) | E (pre-filtro IRT) |")
        md.append("|---|---:|---:|---:|")
        for bud in budgets:
            def cell(pol):
                b = sims[p][pol + "_budget"].get(bud)
                return "—" if not b else f"{b[1]*100:.1f}% @ {b[0]*100:.0f}%"
            md.append(f"| ≤{bud*100:.0f}% | {cell('C')} | {cell('D')} | {cell('E')} |")
        win = sims[p]["winner_text"]
        md.append(f"\n**Vencedor:** {win}\n")
    md.append("## 4. Implicacoes para o app\n")
    md.append("- **Rota por area:** CN e MT devem ser **sempre escaladas** (local ~ acaso); "
              "o ganho por escalar essas areas primeiro e o maior. A confianca so ajuda a "
              "decidir dentro de **LC/CH**.\n")
    md.append("- **Threshold nao deve ser global e alto:** setar `confidenceThresholdHigh=0.98` "
              "(o Youden global) manteria local so os quase-certos e escalaria a grande maioria "
              "— caro. O corte otimo **depende do orcamento** de escalonamento.\n")
    md.append("- **Calibracao:** por causa da superconfianca (ECE alto), o valor bruto de "
              "confianca nao e uma probabilidade de acerto; use-o como *ranking* (via threshold "
              "calibrado por area), nao como probabilidade absoluta.\n")
    md.append("- **IRT confirmado como proxy fraco:** ver correlacoes de Spearman ~ -0.2 no "
              "run anterior; a politica E (pre-filtro IRT) fica atras de C/D.\n")
    (outdir / "relatorio.md").write_text("\n".join(md), encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Analise offline de resultados_acerto_local.csv")
    here = Path(__file__).resolve().parent
    ap.add_argument("--csv", default=str(here / "resultados_acerto_local.csv"))
    ap.add_argument("--out", default=str(here / "analise"))
    ap.add_argument("--prob-esc", default="1.0,0.85",
                    help="prob. de acerto ao escalar (lista). Default 1.0,0.85")
    ap.add_argument("--budgets", default="20,40,60,80",
                    help="orcamentos de escalonamento (%%) p/ comparar. Default 20,40,60,80")
    args = ap.parse_args()

    rows = load(args.csv)
    if not rows:
        print("Nenhuma linha valida no CSV."); return 1
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    p_list = [float(x) for x in args.prob_esc.split(",")]
    budgets = [float(b) / 100.0 for b in args.budgets.split(",")]

    ametrics = area_metrics(rows)
    bins, ece = calibration(rows)

    figs = {"conf": outdir / "fig_conf_dist.png", "roc": outdir / "fig_roc.png",
            "acc": outdir / "fig_acc_area.png", "calib": outdir / "fig_calibracao.png",
            "policy": outdir / "fig_policy_tradeoff.png"}
    fig_conf_dist(rows, figs["conf"])
    fig_roc(rows, ametrics, figs["roc"])
    fig_acc_area(ametrics, figs["acc"])
    fig_calibracao(bins, ece, figs["calib"])

    # sweeps
    ts = list(np.linspace(0, 1.0001, 202))
    diffs = sorted({r["diff"] for r in rows if r["diff"] is not None})
    thetas = list(np.linspace(min(diffs), max(diffs) + 1e-6, 120)) if diffs else []

    sims = {}
    for p in p_list:
        ptsC = sweep(rows, sim_threshold, ts, p)
        ptsD = sweep(rows, sim_area_aware, ts, p)
        ptsE = sweep(rows, sim_difficulty, thetas, p) if thetas else []
        sims[p] = {"C": ptsC, "D": ptsD, "E": ptsE,
                   "C_budget": {}, "D_budget": {}, "E_budget": {}}
        for bud in budgets:
            sims[p]["C_budget"][bud] = best_for_budget(ptsC, bud)
            sims[p]["D_budget"][bud] = best_for_budget(ptsD, bud)
            sims[p]["E_budget"][bud] = best_for_budget(ptsE, bud)
        # vencedor num orcamento intermediario (60% ou o do meio)
        ref = budgets[len(budgets) // 2]
        cand = {k: sims[p][k + "_budget"][ref] for k in ("C", "D", "E")
                if sims[p][k + "_budget"][ref]}
        if cand:
            best_pol = max(cand, key=lambda k: cand[k][1])
            others = ", ".join(f"{k}={cand[k][1]*100:.1f}%" for k in cand)
            sims[p]["winner_text"] = (f"em ≤{ref*100:.0f}% de escalonamento, **politica "
                                      f"{best_pol}** vence ({cand[best_pol][1]*100:.1f}% "
                                      f"@ {cand[best_pol][0]*100:.0f}%). [{others}]")
        else:
            sims[p]["winner_text"] = "n/d"

    # figura de tradeoff usa o primeiro p_esc
    p0 = p_list[0]
    fig_policy(rows, sims[p0]["C"], sims[p0]["D"], sims[p0]["E"], p0, figs["policy"])

    write_reports(outdir, rows, ametrics, bins, ece, sims, p_list, budgets, figs)

    # ── console ──
    print("=" * 70)
    print(f"Questoes: {len(rows)} | Acuracia global: {ametrics['GLOBAL']['acc']*100:.1f}%")
    print(f"ECE (calibracao): {ece:.3f}")
    print("\nPor area:  acc | conf_acerto | conf_erro | AUC")
    for a in AREAS:
        if a in ametrics:
            m = ametrics[a]
            aucs = "n/d" if m["auc"] is None else f"{m['auc']:.3f}"
            print(f"  {a}: {m['acc']*100:5.1f}% | {m['conf_acerto']:.3f} | "
                  f"{m['conf_erro']:.3f} | AUC={aucs}  (n={m['n']})")
    print("\nSimulacao de politicas (acuracia @ escalonamento):")
    for p in p_list:
        print(f"  p_esc={p}:")
        for bud in budgets:
            def c(pol):
                b = sims[p][pol + "_budget"].get(bud)
                return "—" if not b else f"{b[1]*100:.1f}%@{b[0]*100:.0f}%"
            print(f"    <= {int(bud*100)}%: C={c('C')}  D={c('D')}  E={c('E')}")
        print(f"    => {sims[p]['winner_text']}")
    print("\nArquivos em:", outdir)
    for f in figs.values():
        print("  ", f.name)
    print("   relatorio.md, metrics_por_area.csv, calibracao.csv, politicas.csv")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
