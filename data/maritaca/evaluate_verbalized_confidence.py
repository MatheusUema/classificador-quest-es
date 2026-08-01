#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_verbalized_confidence.py — Compara TRÊS sinais de confiança sobre a MESMA
resposta do modelo, para testar se confiança VERBALIZADA e/ou P(True) discriminam
acerto/erro melhor que o LOGPROB (hipótese de Tian et al., EMNLP 2023).

Sinais medidos por questão (mesma resposta de referência = a escolha greedy do MC):
  1) LOGPROB   — baseline: prompt MC, temperature 0, confiança = exp(logprob) do
                 token da letra escolhida (idêntico ao evaluate_local_accuracy.py).
  2) VERBALIZADA — CHAMADA DEDICADA: reusa a letra do MC e faz uma 2ª chamada
                 pedindo SÓ o número 0–100 de confiança na resposta; confiança =
                 numero/100. (Dedicar a chamada reduz muito os faltantes vs pedir
                 letra+numero juntos.)
  3) P(True)   — 2ª chamada perguntando se a resposta proposta está correta
                 (Sim/Não); confiança = probabilidade (exp(logprob)) atribuída a "Sim".

Para CADA sinal reporta: AUC, ECE, confiança média em acerto vs erro e melhor
threshold (Youden). Imprime uma tabela comparando os 3 e salva um CSV
verbalized_vs_logprob_<model>.csv (id, area, acertou, conf_logprob, conf_verbal,
conf_ptrue, ...).

Só stdlib. Reusa utilidades de evaluate_local_accuracy.py (HTTP, parsing de
tokens/letra, load_rows, roc_auc, best_thresholds) e define ECE localmente (stdlib).
NÃO altera o app. NÃO roda sozinho contra o servidor (você sobe o llama-server).

--------------------------------------------------------------------------------
FLUXO (um modelo por vez, mesma porta)
  (a) suba o llama-server apontando pro GGUF do modelo:
        llama-server -m qwen2.5-1.5b-instruct-q4_k_m.gguf --port 8080 --n-probs 10
  (b) rode este script com o rótulo do modelo:
        python evaluate_verbalized_confidence.py --url http://127.0.0.1:8080 --model-name qwen2.5-1.5b
--------------------------------------------------------------------------------
PARSING (documentado)
  - Letra (referência): 1ª letra A–E isolada no texto do MC (ev.extract_letter);
    define acerto vs gabarito. Os três sinais medem confiança SOBRE essa resposta.
  - Verbalizada: parse robusto do 1º número do texto — inteiro 0–100 ("90"->0.90),
    com "%" ("85%"->0.85), com texto ao redor ("Confianca: 90"), fração <=1 com
    decimal ("0.85"/"0,85"->0.85) e truncando >100. Normaliza para [0,1]. Sem número
    válido -> conf_verbal ausente, com motivo ("vazio"/"sem_numero") em `verbal_motivo`
    e o texto cru em `verbal_raw` (auditoria). Faltante não entra na AUC daquele sinal
    (a questão sai da amostra), mas NÃO se inventa valor.
  - P(True): olha-se a distribuição do 1º token gerado na 2ª chamada. Se algum
    candidato for "Sim"/"S"/"Yes"/"True", usa-se a MAIOR prob entre eles. Se o token
    escolhido for "Não"/"No"/"False", usa-se 1 - prob(token). Caso contrário, ausente.
    Requer o servidor com n_probs >= ~10 para expor os candidatos (senão cai no
    fallback do token escolhido).
--------------------------------------------------------------------------------
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import evaluate_local_accuracy as ev   # reuso de utilidades (mesmo diretório)

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "maritaca_enem_irt.csv"
DEFAULT_URL = ev.DEFAULT_URL
LETTERS = ev.LETTERS


# ── Prompts dos três sinais ───────────────────────────────────────────────────
def build_verbalized_prompt(question, alternatives, letra, texto_alt):
    """CHAMADA DEDICADA de confiança verbalizada, reusando a resposta do MC.

    Mostra a questão e a resposta já escolhida (letra + texto) e pede SÓ o número
    0–100. Dedicar a chamada (em vez de pedir letra+numero juntos) reduz muito os
    faltantes, porque o modelo não precisa reformatar a resposta MC."""
    linhas = [question.strip(), ""]
    for i, alt in enumerate(alternatives[:5]):
        linhas.append(f"{LETTERS[i]}) {str(alt).strip()}")
    alt_txt = f"{letra}) {texto_alt}".strip() if texto_alt else f"{letra}"
    linhas += [
        "",
        f"Você respondeu {alt_txt}.",
        "Em uma escala de 0 a 100, qual é a sua confiança de que essa resposta está "
        "correta? Responda APENAS com o número (0 a 100).",
        "",
        "Confiança:",
    ]
    return "\n".join(linhas)


def build_ptrue_prompt(question, alternatives, letra, texto_alt):
    """2ª chamada: a resposta proposta está correta? Sim/Não."""
    linhas = [
        "Considere a questão de múltipla escolha e a resposta proposta abaixo.",
        "A resposta proposta está correta? Responda APENAS com Sim ou Não.",
        "",
        question.strip(),
        "",
    ]
    for i, alt in enumerate(alternatives[:5]):
        linhas.append(f"{LETTERS[i]}) {str(alt).strip()}")
    linhas.append("")
    alt_txt = f"{letra}) {texto_alt}".strip() if texto_alt else f"{letra}"
    linhas.append(f"Resposta proposta: {alt_txt}")
    linhas.append("")
    linhas.append("Está correta (Sim ou Não)?")
    return "\n".join(linhas)


# ── Parsing dos sinais ────────────────────────────────────────────────────────
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def parse_percent(content):
    """Extrai a confiança verbalizada do texto -> (valor em [0,1] ou None, motivo).

    Robusto aos formatos comuns:
      - inteiro 0–100:            "90"            -> 0.90
      - com sinal de porcentagem: "85%"           -> 0.85
      - com texto ao redor:       "Confianca: 90" -> 0.90 (pega o 1º número)
      - fração <= 1 com decimal:  "0.85" / "0,85" -> 0.85 (vírgula normalizada)
      - acima de 100:             "120"           -> trunca em 100 -> 1.00
    motivo ∈ {"ok", "vazio", "sem_numero"}. Não inventa valor: sem número -> None.
    """
    text = (content or "").strip()
    if not text:
        return None, "vazio"
    m = _NUM_RE.search(text)
    if not m:
        return None, "sem_numero"
    tok = m.group(0).replace(",", ".")
    try:
        val = float(tok)
    except ValueError:
        return None, "sem_numero"
    if "." in tok and val <= 1.0:
        conf = val                      # "0.85" -> 0.85 (já em [0,1])
    else:
        if val > 100.0:
            val = 100.0                 # trunca fora da escala
        conf = val / 100.0              # "85" ou "85%" -> 0.85
    conf = max(0.0, min(1.0, conf))
    return conf, "ok"


def _norm_tok(t):
    return "".join(c for c in str(t).strip().lower() if c.isalpha())


_YES = {"sim", "s", "yes", "y", "true", "verdadeiro", "correto", "correta"}
_NO = {"nao", "n", "no", "false", "falso", "incorreto", "incorreta", "errado", "errada"}


def _first_token_dist(cp):
    """Do 1º passo gerado, retorna (chosen_str, chosen_prob, [(cand_str, prob)])."""
    if not cp:
        return None, None, []
    first = cp[0]
    if not isinstance(first, dict):
        return None, None, []
    cands = []
    # schema NOVO com distribuição
    if first.get("top_logprobs"):
        for e in first["top_logprobs"]:
            lp = e.get("logprob")
            cands.append((str(e.get("token", "")),
                          math.exp(float(lp)) if lp is not None else None))
        chosen_str = str(first.get("token", ""))
        clp = first.get("logprob")
        chosen_prob = math.exp(float(clp)) if clp is not None else None
        return chosen_str, chosen_prob, cands
    # schema ANTIGO (probs por candidato)
    if first.get("probs"):
        chosen_str = str(first.get("content", ""))
        for e in first["probs"]:
            cands.append((str(e.get("tok_str", "")), e.get("prob")))
        chosen = next((p for s, p in
                       ((str(e.get("tok_str", "")), e.get("prob")) for e in first["probs"])
                       if s == chosen_str), None)
        return chosen_str, chosen, cands
    # fallback: só o token escolhido (schema novo sem top_logprobs)
    if first.get("logprob") is not None:
        p = math.exp(float(first["logprob"]))
        s = str(first.get("token", ""))
        return s, p, [(s, p)]
    return None, None, []


def ptrue_confidence(resp):
    """Confiança = P(resposta correta) a partir do 1º token (Sim/Não). None se ausente."""
    chosen_str, chosen_prob, cands = _first_token_dist(resp.get("completion_probabilities"))
    # 1) melhor prob entre candidatos "Sim"
    yes_ps = [p for s, p in cands if p is not None and _norm_tok(s) in _YES]
    if yes_ps:
        return max(yes_ps), "sim_dist"
    # 2) token escolhido é "Não" -> P(True) = 1 - prob(Não)
    if chosen_str is not None and chosen_prob is not None:
        if _norm_tok(chosen_str) in _NO:
            return 1.0 - chosen_prob, "nao_compl"
        if _norm_tok(chosen_str) in _YES:
            return chosen_prob, "sim_chosen"
    return None, "ausente"


# ── Métrica ECE (stdlib) ──────────────────────────────────────────────────────
def ece_score(confs, corrects, nbins=10):
    n = len(confs)
    if n == 0:
        return float("nan")
    ece = 0.0
    for b in range(nbins):
        lo, hi = b / nbins, (b + 1) / nbins
        idx = [i for i in range(n)
               if confs[i] >= lo and (confs[i] < hi if b < nbins - 1 else confs[i] <= hi)]
        if not idx:
            continue
        acc_b = sum(corrects[i] for i in idx) / len(idx)
        conf_b = sum(confs[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(acc_b - conf_b)
    return ece


# ── Avaliação por questão (3 chamadas) ────────────────────────────────────────
def eval_row(args, r):
    q, alts = r["question"], r["_alts"]
    label = str(r.get("label", "")).strip().upper()
    out = {"id": r["id"], "ano": r.get("ano", ""), "area": r.get("area", ""),
           "label": label, "escolhido": "?", "acertou": "",
           "conf_logprob": "", "conf_verbal": "", "verbal_motivo": "",
           "verbal_raw": "", "conf_ptrue": "", "ptrue_metodo": ""}

    # 1) LOGPROB (MC greedy) — define a resposta de referência
    try:
        resp1 = ev.completion(args.url, ev.build_mc_prompt(q, alts),
                              n_predict=args.n_predict, n_probs=args.n_probs,
                              temperature=0.0, timeout=args.timeout)
    except Exception as e:
        out["escolhido"] = "ERRO"
        out["_err"] = ev.describe_error(e)
        return out
    tps = ev.token_probs(resp1.get("completion_probabilities"))
    escolhido = ev.extract_letter(resp1.get("content", "")) or "?"
    conf_lp = ev.answer_confidence(tps)
    if conf_lp is None:
        conf_lp = ev.app_confidence(tps)
    out["escolhido"] = escolhido
    out["acertou"] = (escolhido == label)
    out["conf_logprob"] = round(conf_lp, 6) if conf_lp is not None else ""

    # texto da alternativa escolhida (reusado na verbalizada e no P(True))
    idx = LETTERS.find(escolhido) if escolhido in LETTERS else -1
    texto_alt = str(alts[idx]).strip() if 0 <= idx < len(alts) else ""

    # 2) VERBALIZADA — CHAMADA DEDICADA de confiança, reusando a resposta do MC
    try:
        resp2 = ev.completion(args.url,
                              build_verbalized_prompt(q, alts, escolhido, texto_alt),
                              n_predict=args.n_predict_verbal, n_probs=1,
                              temperature=0.0, timeout=args.timeout)
        raw_v = resp2.get("content", "") or ""
        cv, motivo = parse_percent(raw_v)
        out["verbal_raw"] = " ".join(raw_v.split())[:160]
        out["conf_verbal"] = round(cv, 6) if cv is not None else ""
        out["verbal_motivo"] = motivo
    except Exception as e:
        out["conf_verbal"] = ""
        out["verbal_motivo"] = "erro"
        out["_err_verbal"] = ev.describe_error(e)

    # 3) P(True) (2ª chamada Sim/Não sobre a resposta de referência)
    try:
        resp3 = ev.completion(args.url, build_ptrue_prompt(q, alts, escolhido, texto_alt),
                              n_predict=args.n_predict, n_probs=args.n_probs_ptrue,
                              temperature=0.0, timeout=args.timeout)
        cp, metodo = ptrue_confidence(resp3)
        out["conf_ptrue"] = round(cp, 6) if cp is not None else ""
        out["ptrue_metodo"] = metodo
    except Exception as e:
        out["conf_ptrue"] = ""
        out["ptrue_metodo"] = "erro"
        out["_err_ptrue"] = ev.describe_error(e)
    return out


# ── Métricas por sinal ────────────────────────────────────────────────────────
def signal_metrics(results, key):
    pairs = [(float(x[key]), 1 if x["acertou"] is True else 0)
             for x in results
             if x["acertou"] in (True, False) and x.get(key) not in ("", None)]
    n_valid = sum(1 for x in results if x["acertou"] in (True, False))
    m = {"n": len(pairs), "faltantes": n_valid - len(pairs),
         "auc": None, "ece": float("nan"),
         "conf_acerto": float("nan"), "conf_erro": float("nan"), "thr": None}
    if len(pairs) < 2:
        return m
    scores = [s for s, _ in pairs]
    labels = [l for _, l in pairs]
    ca = [s for s, l in pairs if l == 1]
    ce = [s for s, l in pairs if l == 0]
    m["auc"] = ev.roc_auc(scores, labels)
    m["ece"] = ece_score(scores, labels)
    m["conf_acerto"] = sum(ca) / len(ca) if ca else float("nan")
    m["conf_erro"] = sum(ce) / len(ce) if ce else float("nan")
    bt = ev.best_thresholds(scores, labels)
    if bt:
        m["thr"] = bt[2]   # threshold de Youden
    return m


SIGNALS = [("conf_logprob", "LOGPROB"), ("conf_verbal", "VERBALIZADA"), ("conf_ptrue", "P(True)")]


def fmt(v, kind):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if kind == "f3":
        return f"{v:.3f}"
    return str(v)


def print_table(metrics):
    cols = [name for _, name in SIGNALS]
    w = 13
    head = "metrica".ljust(14) + " | " + " | ".join(c.rjust(w) for c in cols)
    print("=" * len(head))
    print("COMPARATIVO DE SINAIS DE CONFIANCA (mesma resposta)")
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    rows = [("AUC (disc.)", "auc", "f3"), ("ECE (calib.)", "ece", "f3"),
            ("Conf. acerto", "conf_acerto", "f3"), ("Conf. erro", "conf_erro", "f3"),
            ("Thr (Youden)", "thr", "f3"), ("N usados", "n", "int"),
            ("Faltantes", "faltantes", "int")]
    for label, key, kind in rows:
        cells = " | ".join(fmt(metrics[s].get(key), kind).rjust(w) for s, _ in SIGNALS)
        print(label.ljust(14) + " | " + cells)
    print("=" * len(head))
    best = max((s for s, _ in SIGNALS if metrics[s]["auc"] is not None),
               key=lambda s: metrics[s]["auc"], default=None)
    if best:
        nm = dict(SIGNALS)[best]
        print(f">> Maior AUC: {nm} ({metrics[best]['auc']:.3f}). "
              f"Se VERBALIZADA/P(True) > LOGPROB, apoia Tian et al. (2023).")


def verbal_reasons(results):
    """Contagem dos motivos de faltante/sucesso da verbalizada (para auditoria)."""
    counts = {}
    for x in results:
        if x["acertou"] not in (True, False):
            continue
        counts[x.get("verbal_motivo", "")] = counts.get(x.get("verbal_motivo", ""), 0) + 1
    return counts


def save_csv(path, results):
    cols = ["id", "ano", "area", "escolhido", "label", "acertou",
            "conf_logprob", "conf_verbal", "verbal_motivo", "verbal_raw",
            "conf_ptrue", "ptrue_metodo"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(results)


# ── Main ──────────────────────────────────────────────────────────────────────
def run(args):
    if args.model_name.strip() and not args.out:
        args.out = str(HERE / f"verbalized_vs_logprob_{ev.safe_label(args.model_name)}.csv")
    elif not args.out:
        args.out = str(HERE / "verbalized_vs_logprob.csv")
    # atributos exigidos por ev.load_rows
    args.include_images = False
    args.require_irt = True
    rows = ev.load_rows(args)

    print(f"Dataset: {args.csv}")
    if args.model_name.strip():
        print(f"Modelo (rotulo): {args.model_name.strip()}")
    print(f"Servidor: {args.url} | questoes: {len(rows)} "
          f"(IU=false, has_irt{', areas=' + args.areas if args.areas else ''})")
    print("Cada questao faz 3 chamadas: MC (logprob), verbalizada, P(True).")
    print("-" * 72)

    results = []
    n_err = 0
    for k, r in enumerate(rows, 1):
        out = eval_row(args, r)
        if out["escolhido"] == "ERRO":
            n_err += 1
            print(f"[{k}/{len(rows)}] {r['id']} ERRO MC: {out.get('_err', '')}")
        results.append(out)
        if k % 10 == 0 or k == len(rows):
            acc = sum(1 for x in results if x["acertou"] is True)
            print(f"[{k}/{len(rows)}] parcial: acertos={acc}")

    save_csv(args.out, results)
    print(f"\nResultados por questao: {args.out}")

    metrics = {key: signal_metrics(results, key) for key, _ in SIGNALS}
    print()
    print_table(metrics)
    reasons = verbal_reasons(results)
    faltou = {k: v for k, v in reasons.items() if k not in ("ok",)}
    print(f"Verbalizada — capturados: {reasons.get('ok', 0)} | "
          f"faltantes por motivo: {faltou if faltou else '{}'}")
    if n_err:
        print(f"(erros de rede no MC: {n_err})")


def main():
    ap = argparse.ArgumentParser(
        description="Compara LOGPROB vs VERBALIZADA vs P(True) como confianca em MCQ.")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"default {DEFAULT_URL}")
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="dataset unificado")
    ap.add_argument("--model-name", dest="model_name", default="",
                    help="rotulo do modelo; nomeia a saida verbalized_vs_logprob_<rotulo>.csv")
    ap.add_argument("--out", default=None, help="CSV de saida (default por --model-name)")
    ap.add_argument("--limit", type=int, default=0, help="avalia so as N primeiras (0=todas)")
    ap.add_argument("--areas", default="", help="filtra areas, ex.: LC,CH,CN,MT")
    ap.add_argument("--timeout", type=float, default=300.0, help="timeout (s) por chamada")
    ap.add_argument("--n-predict", type=int, default=4,
                    help="tokens no MC e no P(True). Default 4")
    ap.add_argument("--n-predict-verbal", type=int, default=16,
                    help="tokens na geracao verbalizada (letra + numero). Default 16")
    ap.add_argument("--n-probs", type=int, default=5, help="n_probs do MC. Default 5")
    ap.add_argument("--n-probs-ptrue", type=int, default=10,
                    help="n_probs do P(True) (precisa expor 'Sim' na distribuicao). Default 10")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
