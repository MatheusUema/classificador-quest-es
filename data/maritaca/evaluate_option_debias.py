#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_option_debias.py — Debiasing / robustez de OPÇÃO em MCQ.

Objetivo: checar (i) o VIÉS DE POSIÇÃO/SELEÇÃO do modelo (ele prefere certas letras?)
e (ii) se o SINAL DE CONFIANÇA (logprob) é ROBUSTO à ordem das alternativas. É a
guarda contra a objeção de revisor de Zheng et al. (ICLR 2024): "LLMs não são
seletores de múltipla escolha robustos".

Método (K = 5 PERMUTAÇÕES CÍCLICAS por questão):
  As 5 alternativas são rotacionadas por 1 a cada permutação, de modo que a
  alternativa CORRETA ocupe cada posição A–E EXATAMENTE UMA VEZ (perm_idx 0..4).
  Em cada permutação: monta o prompt MC (temperature 0), lê a LETRA escolhida
  (posição), mapeia de volta para qual ALTERNATIVA (conteúdo) foi escolhida, se
  acertou, e a confiança logprob = exp(logprob) do token da letra escolhida.

Rotação: perm_alts[pos] = alts[(pos + s) % 5] para a permutação s.
  -> a correta (índice de conteúdo c) fica na posição (c - s) % 5.
  -> o conteúdo escolhido a partir da posição p é (p + s) % 5.

Saídas:
  - CSV por (questao, permutacao): resultados_debias_<model>.csv
    (id, ano, area, label, perm_idx, pos_correta, letra_escolhida,
     conteudo_escolhido_id, acertou, conf_logprob)
  - Resumo impresso + debias_resumo_<model>.csv com:
      1. VIÉS DE POSIÇÃO: frequência de escolha por posição A–E (ideal ~20% cada),
         com desvio max-min e qui-quadrado simples.
      2. ACURÁCIA por POSIÇÃO DA CORRETA (a corretude depende de onde está a certa?).
      3. ROBUSTEZ DO SINAL: AUC do logprob prevendo acerto, AGREGADO sobre as 5
         permutações, comparado ao baseline de ordem única.
      4. ACURÁCIA DEBIASED: voto majoritário do CONTEÚDO escolhido nas 5 permutações
         vs acurácia de ordem única; e CONSISTÊNCIA (fração de questões em que o
         mesmo conteúdo é escolhido em >= k das 5 permutações).

Só stdlib. Reusa utilidades de evaluate_local_accuracy.py (HTTP, prompt MC, parsing de
letra/confiança, load_rows, roc_auc). NÃO altera o app. NÃO roda sozinho contra o
servidor.

--------------------------------------------------------------------------------
CUSTO: 5 chamadas por questão. Rode primeiro um SANITY com --limit 40 e depois o
conjunto completo (389). Foco no frontrunner Qwen2.5-1.5B.

  (a) suba o servidor:
        llama-server -m qwen2.5-1.5b-instruct-q4_k_m.gguf --port 8080 --n-probs 5
  (b) sanity (40 questoes):
        python evaluate_option_debias.py --url http://127.0.0.1:8080 --model-name qwen2.5-1.5b --limit 40
  (c) completo:
        python evaluate_option_debias.py --url http://127.0.0.1:8080 --model-name qwen2.5-1.5b
--------------------------------------------------------------------------------
"""

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

import evaluate_local_accuracy as ev   # reuso de utilidades (mesmo diretório)

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "maritaca_enem_irt.csv"
DEFAULT_URL = ev.DEFAULT_URL
LETTERS = ev.LETTERS
K = 5   # permutações cíclicas (uma por posição da correta)


# ── Avaliação de uma questão sob as 5 permutações ─────────────────────────────
def eval_question(args, r):
    """Retorna lista de dicts (um por permutação) para a questão r."""
    alts = [str(a).strip() for a in r["_alts"][:5]]
    label = str(r.get("label", "")).strip().upper()
    c = LETTERS.find(label)                 # índice de conteúdo da alternativa correta
    if c < 0 or len(alts) < 5:
        return []
    linhas = []
    for s in range(K):
        perm_alts = [alts[(pos + s) % 5] for pos in range(5)]
        pos_correta = (c - s) % 5           # onde a correta caiu nesta permutação
        rec = {"id": r["id"], "ano": r.get("ano", ""), "area": r.get("area", ""),
               "label": label, "perm_idx": s, "pos_correta": LETTERS[pos_correta],
               "letra_escolhida": "", "conteudo_escolhido_id": "",
               "acertou": "", "conf_logprob": ""}
        try:
            resp = ev.completion(args.url, ev.build_mc_prompt(r["question"], perm_alts),
                                 n_predict=args.n_predict, n_probs=args.n_probs,
                                 temperature=0.0, timeout=args.timeout)
        except Exception as e:
            rec["letra_escolhida"] = "ERRO"
            rec["_err"] = ev.describe_error(e)
            linhas.append(rec)
            continue
        tps = ev.token_probs(resp.get("completion_probabilities"))
        letra = ev.extract_letter(resp.get("content", ""))
        conf = ev.answer_confidence(tps)
        if conf is None:
            conf = ev.app_confidence(tps)
        if letra and letra in LETTERS:
            p = LETTERS.find(letra)
            content_id = (p + s) % 5        # de volta ao conteúdo original
            rec["letra_escolhida"] = letra
            rec["conteudo_escolhido_id"] = content_id
            rec["acertou"] = (content_id == c)
        else:
            rec["letra_escolhida"] = "?"
            rec["acertou"] = False          # sem letra = erro
        rec["conf_logprob"] = round(conf, 6) if conf is not None else ""
        linhas.append(rec)
    return linhas


# ── Métricas ──────────────────────────────────────────────────────────────────
def position_bias(rows):
    """Distribuição das POSIÇÕES (letras) escolhidas + qui-quadrado simples."""
    chosen = [x["letra_escolhida"] for x in rows if x["letra_escolhida"] in LETTERS]
    n = len(chosen)
    cnt = Counter(chosen)
    dist = {L: cnt.get(L, 0) for L in LETTERS}
    pct = {L: (dist[L] / n if n else 0.0) for L in LETTERS}
    exp = n / 5.0 if n else 0.0
    chi2 = sum((dist[L] - exp) ** 2 / exp for L in LETTERS) if exp else float("nan")
    span = (max(pct.values()) - min(pct.values())) if n else float("nan")
    return n, dist, pct, chi2, span


def acc_by_correct_position(rows):
    """Acurácia condicionada à POSIÇÃO em que a correta foi colocada."""
    out = {}
    for L in LETTERS:
        sub = [x for x in rows if x["pos_correta"] == L and x["acertou"] in (True, False)]
        out[L] = (sum(1 for x in sub if x["acertou"]) / len(sub), len(sub)) if sub else (float("nan"), 0)
    return out


def aggregate_auc(rows):
    """AUC do logprob prevendo acerto, sobre TODAS as permutações."""
    pairs = [(float(x["conf_logprob"]), 1 if x["acertou"] is True else 0)
             for x in rows if x["acertou"] in (True, False) and x["conf_logprob"] != ""]
    if len(pairs) < 2:
        return None, 0
    return ev.roc_auc([s for s, _ in pairs], [l for _, l in pairs]), len(pairs)


def debias_vote(rows, kmin):
    """Voto majoritário do CONTEÚDO escolhido por questão -> acurácia debiased +
    acurácia de ordem única (perm 0) + consistência (>= kmin iguais das 5)."""
    byq = {}
    for x in rows:
        byq.setdefault(x["id"], []).append(x)
    n_deb = correct_deb = 0
    n_single = correct_single = 0
    consist_hist = Counter()      # maior contagem de um mesmo conteúdo por questão
    consist_ge = 0
    for qid, recs in byq.items():
        c = LETTERS.find(recs[0]["label"])
        contents = [x["conteudo_escolhido_id"] for x in recs
                    if isinstance(x["conteudo_escolhido_id"], int)]
        # ordem única = permutação 0
        p0 = next((x for x in recs if x["perm_idx"] == 0), None)
        if p0 is not None and p0["acertou"] in (True, False):
            n_single += 1
            correct_single += 1 if p0["acertou"] else 0
        if not contents:
            continue
        cnt = Counter(contents)
        top_content, top_n = cnt.most_common(1)[0]
        consist_hist[top_n] += 1
        if top_n >= kmin:
            consist_ge += 1
        n_deb += 1
        correct_deb += 1 if top_content == c else 0
    acc_deb = correct_deb / n_deb if n_deb else float("nan")
    acc_single = correct_single / n_single if n_single else float("nan")
    frac_consist = consist_ge / n_deb if n_deb else float("nan")
    return {"acc_debiased": acc_deb, "n_debiased": n_deb,
            "acc_single": acc_single, "n_single": n_single,
            "consist_hist": dict(consist_hist), "frac_consist_ge": frac_consist, "kmin": kmin}


# ── Impressão / persistência ──────────────────────────────────────────────────
def print_summary(rows, kmin):
    n_pos, dist, pct, chi2, span = position_bias(rows)
    accpos = acc_by_correct_position(rows)
    auc, n_auc = aggregate_auc(rows)
    deb = debias_vote(rows, kmin)

    print("=" * 66)
    print("DEBIASING / ROBUSTEZ DE OPCAO (5 permutacoes ciclicas)")
    print("=" * 66)
    print(f"Chamadas com letra valida: {n_pos}")

    print("\n1) VIES DE POSICAO — quantas vezes cada LETRA foi escolhida (ideal ~20%):")
    for L in LETTERS:
        print(f"   {L}: {dist[L]:4d}  ({pct[L]*100:5.1f}%)")
    print(f"   desvio max-min: {span*100:.1f} pts | qui-quadrado (df=4): {chi2:.1f}"
          f"   (chi2 alto => vies de posicao)")

    print("\n2) ACURACIA por POSICAO DA CORRETA (assinatura de vies se desigual):")
    for L in LETTERS:
        a, nn = accpos[L]
        aa = "n/d" if (isinstance(a, float) and math.isnan(a)) else f"{a*100:5.1f}%"
        print(f"   correta em {L}: {aa}  (n={nn})")

    print("\n3) ROBUSTEZ DO SINAL — AUC do logprob (agregado nas 5 permutacoes):")
    aucs = "n/d" if auc is None else f"{auc:.3f}"
    print(f"   AUC agregada = {aucs}  (n={n_auc})")
    print("   baseline de ordem unica (Qwen2.5-1.5B, doc 03 §5): 0,793.")
    print("   Se a AUC agregada ~ 0,79, o sinal e ROBUSTO a ordem das alternativas.")

    print("\n4) ACURACIA DEBIASED (voto majoritario do CONTEUDO nas 5 permutacoes):")
    print(f"   ordem unica (perm 0): {deb['acc_single']*100:.1f}%  (n={deb['n_single']})")
    print(f"   debiased (voto 5):    {deb['acc_debiased']*100:.1f}%  (n={deb['n_debiased']})")
    print(f"   consistencia (mesmo conteudo em >= {kmin} das 5): "
          f"{deb['frac_consist_ge']*100:.1f}% das questoes")
    print(f"   histograma de concordancia (max iguais das 5): {deb['consist_hist']}")
    print("=" * 66)
    return {"n_pos": n_pos, "dist": dist, "pct": pct, "chi2": chi2, "span": span,
            "accpos": accpos, "auc": auc, "n_auc": n_auc, "deb": deb}


def save_detail_csv(path, rows):
    cols = ["id", "ano", "area", "label", "perm_idx", "pos_correta",
            "letra_escolhida", "conteudo_escolhido_id", "acertou", "conf_logprob"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)


def save_summary_csv(path, S, kmin):
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["metrica", "valor"])
        for L in LETTERS:
            wr.writerow([f"freq_pos_{L}", f"{S['pct'][L]:.4f}"])
        wr.writerow(["pos_span_maxmin", f"{S['span']:.4f}"])
        wr.writerow(["pos_qui_quadrado", f"{S['chi2']:.4f}"])
        for L in LETTERS:
            a, nn = S["accpos"][L]
            wr.writerow([f"acc_correta_em_{L}", "" if math.isnan(a) else f"{a:.4f}"])
        wr.writerow(["auc_logprob_agregada", "" if S["auc"] is None else f"{S['auc']:.4f}"])
        wr.writerow(["n_auc", S["n_auc"]])
        wr.writerow(["acc_ordem_unica", f"{S['deb']['acc_single']:.4f}"])
        wr.writerow(["acc_debiased_voto5", f"{S['deb']['acc_debiased']:.4f}"])
        wr.writerow([f"frac_consistencia_ge_{kmin}", f"{S['deb']['frac_consist_ge']:.4f}"])


# ── Main ──────────────────────────────────────────────────────────────────────
def run(args):
    tag = ev.safe_label(args.model_name) if args.model_name.strip() else "local"
    if not args.out:
        args.out = str(HERE / f"resultados_debias_{tag}.csv")
    summary_out = str(HERE / f"debias_resumo_{tag}.csv")
    args.include_images = False
    args.require_irt = True
    qs = ev.load_rows(args)

    print(f"Dataset: {args.csv}")
    if args.model_name.strip():
        print(f"Modelo (rotulo): {args.model_name.strip()}")
    print(f"Servidor: {args.url} | questoes: {len(qs)} x {K} permutacoes = "
          f"{len(qs) * K} chamadas (IU=false, has_irt"
          f"{', areas=' + args.areas if args.areas else ''})")
    print("-" * 66)

    rows = []
    n_err = 0
    for k, r in enumerate(qs, 1):
        recs = eval_question(args, r)
        for rec in recs:
            if rec["letra_escolhida"] == "ERRO":
                n_err += 1
        rows.extend(recs)
        if k % 10 == 0 or k == len(qs):
            print(f"[{k}/{len(qs)}] questoes processadas ({len(rows)} linhas)")

    save_detail_csv(args.out, rows)
    print(f"\nDetalhe por (questao, permutacao): {args.out}")
    S = print_summary(rows, args.kmin)
    save_summary_csv(summary_out, S, args.kmin)
    print(f"Resumo: {summary_out}")
    if n_err:
        print(f"(erros de rede: {n_err} chamadas)")


def main():
    ap = argparse.ArgumentParser(
        description="Debiasing/robustez de opcao em MCQ (5 permutacoes ciclicas).")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"default {DEFAULT_URL}")
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="dataset unificado")
    ap.add_argument("--model-name", dest="model_name", default="",
                    help="rotulo do modelo; nomeia resultados_debias_<rotulo>.csv")
    ap.add_argument("--out", default=None, help="CSV detalhado de saida (default por --model-name)")
    ap.add_argument("--limit", type=int, default=0, help="avalia so as N primeiras (0=todas)")
    ap.add_argument("--areas", default="", help="filtra areas, ex.: LC,CH,CN,MT")
    ap.add_argument("--timeout", type=float, default=300.0, help="timeout (s) por chamada")
    ap.add_argument("--n-predict", type=int, default=4, help="tokens gerados no MC. Default 4")
    ap.add_argument("--n-probs", type=int, default=5)
    ap.add_argument("--kmin", type=int, default=3,
                    help="k para a consistencia (mesmo conteudo em >= k das 5). Default 3")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
