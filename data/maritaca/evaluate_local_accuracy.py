#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_local_accuracy.py — Calibra os thresholds de confiança do tier servidor
usando o rótulo CERTO: o ACERTO do modelo (comparando com o gabarito), e não a
dificuldade IRT (que mede dificuldade para HUMANOS, proxy fraco para a LLM — doc 01).

Fluxo:
  1. Lê o dataset unificado maritaca_enem_irt.csv (question, alternatives, label, ...).
  2. Monta um prompt de múltipla escolha (PT-BR, alternativas A–E, "responda só a letra").
  3. Consulta o llama-server /completion (com n_probs) — mesma lógica do server_smoke_test.py.
  4. Extrai a letra escolhida e a CONFIANCA (prob do token da letra, via exp(logprob);
     fallback: média de exp(logprob) dos tokens gerados = método do app).
  5. Compara com o gabarito e reporta: acurácia (global/área), confiança em ACERTO vs ERRO,
     melhor threshold (varredura + Youden J + ROC AUC), e correlações de Spearman
     (IRT×confiança e IRT×acerto) para documentar que o IRT prevê mal a dificuldade-LLM.
  6. Salva resultados_acerto_local.csv e imprime o sumário.

Só stdlib (urllib, csv, json, math). NÃO altera o app. NÃO roda sozinho contra o
servidor além do que você mandar (você executa; nada é chamado no import).

--------------------------------------------------------------------------------
MODO MULTI-MODELO (avaliar várias LLMs locais, uma de cada vez)
  O usuário compara 4 modelos locais servidos pelo llama.cpp, um por vez, na MESMA
  porta (ex.: 8080). Para CADA modelo:
    (a) suba o llama-server apontando para o GGUF daquele modelo na porta 8080, p.ex.:
          llama-server -m gemma-3-1b-it-Q4_K_M.gguf   --port 8080 --n-probs 5
          llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf --port 8080 --n-probs 5
    (b) com o servidor no ar, rode este script passando um rótulo em --model-name:
          python evaluate_local_accuracy.py --url http://127.0.0.1:8080 --model-name gemma-3-1b
          python evaluate_local_accuracy.py --url http://127.0.0.1:8080 --model-name qwen2.5-0.5b
        Isso grava a coluna `model=<rótulo>` em cada linha e, se --out não for dado,
        salva em resultados_acerto_<model-name>.csv (um CSV por modelo).
    (c) ao terminar os 4 modelos, rode UMA vez aggregate_multimodel.py para juntar
        todos os resultados_acerto_*.csv num comparativo (ver aquele script).
  O protocolo de avaliação NÃO muda entre modelos (mesmo prompt MC, temperature 0,
  n_predict 4, mesma métrica de confiança) — só o rótulo e o arquivo de saída mudam.
--------------------------------------------------------------------------------
PARSING DA RESPOSTA (documentado)
  - Letra escolhida: primeira letra A–E "isolada" (sem letra colada antes/depois) no
    texto `content`; se não houver, a primeira letra A–E qualquer; se nenhuma, conta-se
    como NAO-EXTRAIDA (escolhido="?", acertou=False).
  - Confiança: procura o 1º token gerado que seja uma letra A–E e usa exp(logprob) dele
    (= probabilidade que o modelo deu à letra respondida). Se não achar, usa a média de
    exp(logprob) de todos os tokens gerados (idêntico ao calculateConfidence do app).
    A coluna `conf_metodo` indica qual foi usada ("letra" ou "media_app").
  - Suporta os dois formatos de completion_probabilities (novo token/logprob e antigo
    tok_str/prob), como o app.
--------------------------------------------------------------------------------
"""

import argparse
import csv
import json
import math
import re
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "maritaca_enem_irt.csv"
DEFAULT_OUT = HERE / "resultados_acerto_local.csv"
DEFAULT_URL = "http://192.168.1.100:8080"   # mesmo default do ServerConfig do app
LETTERS = "ABCDE"


def safe_label(name):
    """Rótulo de modelo -> fragmento de nome de arquivo seguro (Windows/Unix)."""
    keep = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in name.strip())
    return keep.strip("-") or "modelo"


def resolve_out(args):
    """Resolve o caminho de saída: --out explícito vence; senão usa o rótulo do modelo."""
    if args.out:
        return Path(args.out)
    if args.model_name.strip():
        return HERE / f"resultados_acerto_{safe_label(args.model_name)}.csv"
    return DEFAULT_OUT


# ── HTTP (mesma base do server_smoke_test.py) ────────────────────────────────
def completion(base_url, prompt, n_predict, n_probs, temperature, timeout):
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 40,
        "n_probs": n_probs,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/completion", data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def describe_error(e):
    if isinstance(e, urllib.error.HTTPError):
        return f"HTTP {e.code} {e.reason}"
    if isinstance(e, urllib.error.URLError):
        r = e.reason
        if isinstance(r, (socket.timeout, TimeoutError)):
            return "TIMEOUT"
        if isinstance(r, ConnectionRefusedError) or "refused" in str(r).lower():
            return "CONEXAO RECUSADA"
        return f"REDE: {r}"
    if isinstance(e, (socket.timeout, TimeoutError)):
        return "TIMEOUT"
    if isinstance(e, ConnectionRefusedError):
        return "CONEXAO RECUSADA"
    return f"{type(e).__name__}: {e}"


# ── Extração de tokens/prob (novo e antigo schema) ───────────────────────────
def token_probs(cp):
    """Lista [(token_str, prob)] por posição gerada. prob = exp(logprob) no schema novo."""
    out = []
    if not cp:
        return out
    for tp in cp:
        if not isinstance(tp, dict):
            continue
        if tp.get("logprob") is not None:                      # schema NOVO
            out.append((str(tp.get("token", "")), math.exp(float(tp["logprob"]))))
        elif tp.get("probs"):                                  # schema ANTIGO
            content = tp.get("content", "")
            probs = tp["probs"]
            chosen = next((e for e in probs if e.get("tok_str") == content), None) or probs[0]
            p = chosen.get("prob")
            if p is not None:
                out.append((str(content), float(p)))
    return out


def is_letter_token(tok):
    """True e retorna a letra se o token contém exatamente uma letra A–E e nada de outras letras."""
    t = tok.strip().upper()
    letters = [c for c in t if c.isalpha()]
    if len(letters) == 1 and letters[0] in LETTERS:
        return letters[0]
    return None


def extract_letter(content):
    m = re.search(r"(?<![A-Za-z])([A-Ea-e])(?![A-Za-z])", content)
    if m:
        return m.group(1).upper()
    m = re.search(r"[A-Ea-e]", content)
    return m.group(0).upper() if m else None


def app_confidence(tps):
    """calculateConfidence do app: média de exp(logprob) dos tokens gerados."""
    vals = [p for _, p in tps]
    return sum(vals) / len(vals) if vals else -1.0


def answer_confidence(tps):
    """Prob do 1º token-letra (confiança na letra respondida); None se não houver."""
    for tok, p in tps:
        if is_letter_token(tok):
            return p
    return None


# ── Prompt de múltipla escolha (PT-BR) ───────────────────────────────────────
def build_mc_prompt(question, alternatives):
    linhas = [
        "Responda à questão de múltipla escolha abaixo. Escolha a única alternativa "
        "correta e responda APENAS com a letra (A, B, C, D ou E), sem explicação.",
        "",
        question.strip(),
        "",
    ]
    for i, alt in enumerate(alternatives[:5]):
        linhas.append(f"{LETTERS[i]}) {str(alt).strip()}")
    linhas.append("")
    linhas.append("Resposta:")
    return "\n".join(linhas)


# ── Estatística (stdlib) ─────────────────────────────────────────────────────
def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0     # ranks 1-based, média em empates
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(a, b):
    n = len(a)
    if n < 2:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((x - mb) ** 2 for x in b))
    return num / (da * db) if da and db else None


def spearman(a, b):
    if len(a) < 2:
        return None
    return pearson(rankdata(a), rankdata(b))


def roc_auc(scores, positive):
    """AUC = P(score_correto > score_errado). positive: lista 0/1 (1 = acerto)."""
    ranks = rankdata(scores)
    pos = [ranks[i] for i in range(len(scores)) if positive[i] == 1]
    npos, nneg = len(pos), len(scores) - sum(positive)
    if npos == 0 or nneg == 0:
        return None
    return (sum(pos) - npos * (npos + 1) / 2.0) / (npos * nneg)


def best_thresholds(scores, positive):
    """Varre limiares: retorna (t_acuracia, acc, t_youden, J, tpr, fpr)."""
    npos, nneg = sum(positive), len(positive) - sum(positive)
    if npos == 0 or nneg == 0:
        return None
    cands = sorted(set(scores))
    best_acc = (-1, None)
    best_j = (-1, None, 0, 0)
    for t in cands:
        tp = sum(1 for i in range(len(scores)) if scores[i] >= t and positive[i] == 1)
        fp = sum(1 for i in range(len(scores)) if scores[i] >= t and positive[i] == 0)
        tn = nneg - fp
        acc = (tp + tn) / len(scores)
        tpr = tp / npos
        fpr = fp / nneg
        j = tpr - fpr
        if acc > best_acc[0]:
            best_acc = (acc, t)
        if j > best_j[0]:
            best_j = (j, t, tpr, fpr)
    return best_acc[1], best_acc[0], best_j[1], best_j[0], best_j[2], best_j[3]


# ── Pipeline ─────────────────────────────────────────────────────────────────
def load_rows(args):
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    areas = set(a.strip().upper() for a in args.areas.split(",")) if args.areas else None
    sel = []
    for r in rows:
        if not args.include_images and str(r.get("IU", "")).strip().lower() == "true":
            continue
        if args.require_irt and str(r.get("has_irt", "")).strip().lower() != "true":
            continue
        if areas and r.get("area", "").upper() not in areas:
            continue
        try:
            r["_alts"] = json.loads(r.get("alternatives") or "[]")
        except json.JSONDecodeError:
            r["_alts"] = []
        if len(r["_alts"]) < 5 or not str(r.get("label", "")).strip():
            continue
        sel.append(r)
    if args.limit:
        sel = sel[:args.limit]
    return sel


def run(args):
    args.out = str(resolve_out(args))
    model_name = args.model_name.strip()
    rows = load_rows(args)
    print(f"Dataset: {args.csv}")
    if model_name:
        print(f"Modelo (rotulo): {model_name}")
    print(f"Servidor: {args.url}  (timeout={args.timeout:.0f}s, n_predict={args.n_predict}, "
          f"temperature={args.temperature})")
    print(f"Questoes selecionadas: {len(rows)} "
          f"(IU=false={'nao' if args.include_images else 'sim'}, "
          f"require_irt={'sim' if args.require_irt else 'nao'}"
          f"{', areas=' + args.areas if args.areas else ''})")
    print("-" * 72)

    results = []
    n_err = n_noletter = 0
    for k, r in enumerate(rows, 1):
        prompt = build_mc_prompt(r["question"], r["_alts"])
        try:
            resp = completion(args.url, prompt, args.n_predict, args.n_probs,
                              args.temperature, args.timeout)
        except Exception as e:
            n_err += 1
            print(f"[{k}/{len(rows)}] {r['id']} ERRO: {describe_error(e)}")
            results.append({**base_row(r, args.n_predict, model_name), "escolhido": "ERRO",
                            "acertou": "", "confianca": "", "conf_metodo": "erro",
                            "conf_app": ""})
            continue
        content = resp.get("content", "")
        tps = token_probs(resp.get("completion_probabilities"))
        escolhido = extract_letter(content)
        conf_letter = answer_confidence(tps)
        conf_app = app_confidence(tps)
        if conf_letter is not None:
            conf, metodo = conf_letter, "letra"
        else:
            conf, metodo = conf_app, "media_app"
        if escolhido is None:
            n_noletter += 1
            escolhido = "?"
        label = str(r["label"]).strip().upper()
        acertou = (escolhido == label)
        results.append({**base_row(r, args.n_predict, model_name), "escolhido": escolhido, "acertou": acertou,
                        "confianca": round(conf, 6) if conf is not None else "",
                        "conf_metodo": metodo,
                        "conf_app": round(conf_app, 6) if conf_app is not None else ""})
        if k % 10 == 0 or k == len(rows):
            print(f"[{k}/{len(rows)}] parcial: acertos="
                  f"{sum(1 for x in results if x['acertou'] is True)}")

    save_csv(args.out, results)
    print(f"\nResultados por questao: {args.out}")
    summarize(results, n_err, n_noletter)


def base_row(r, n_predict, model_name=""):
    return {
        "model": model_name,
        "id": r["id"], "ano": r.get("ano", ""), "area": r.get("area", ""),
        "difficulty_score": r.get("difficulty_score", ""),
        "label": str(r.get("label", "")).strip().upper(),
        "n_predict": n_predict,
    }


def save_csv(path, results):
    cols = ["model", "id", "ano", "area", "difficulty_score", "escolhido", "label",
            "acertou", "confianca", "conf_metodo", "conf_app", "n_predict"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(results)


def summarize(results, n_err, n_noletter):
    val = [x for x in results if x["acertou"] in (True, False)]
    if not val:
        print("\nNenhuma questao valida (todas com erro?). Sem sumario.")
        return
    acc = sum(1 for x in val if x["acertou"]) / len(val)
    print("\n==================== SUMARIO ====================")
    print(f"Questoes avaliadas: {len(val)} | erros de rede: {n_err} | sem letra extraida: {n_noletter}")
    print(f"ACURACIA local global: {acc:.1%}  ({sum(1 for x in val if x['acertou'])}/{len(val)})")

    print("\nAcuracia por area:")
    for a in ("LC", "CH", "CN", "MT"):
        sub = [x for x in val if x["area"] == a]
        if sub:
            print(f"  {a}: {sum(1 for x in sub if x['acertou'])}/{len(sub)} = "
                  f"{sum(1 for x in sub if x['acertou'])/len(sub):.1%}")

    conf = [(float(x["confianca"]), 1 if x["acertou"] else 0)
            for x in val if x["confianca"] != ""]
    if len(conf) < 2:
        print("\n(Confianca insuficiente para as metricas de separacao.)")
        return
    scores = [c for c, _ in conf]
    labels = [l for _, l in conf]
    c_acerto = [c for c, l in conf if l == 1]
    c_erro = [c for c, l in conf if l == 0]

    print("\n>>> A confianca separa ACERTO de ERRO? (ponto central)")
    if c_acerto:
        print(f"  confianca media em ACERTOS: {sum(c_acerto)/len(c_acerto):.3f} "
              f"(mediana {median(c_acerto):.3f}, n={len(c_acerto)})")
    if c_erro:
        print(f"  confianca media em ERROS  : {sum(c_erro)/len(c_erro):.3f} "
              f"(mediana {median(c_erro):.3f}, n={len(c_erro)})")

    auc = roc_auc(scores, labels)
    if auc is not None:
        print(f"  ROC AUC (confianca prevendo acerto): {auc:.3f}  "
              f"(0.5=aleatorio; >0.7 = sinal util)")
    bt = best_thresholds(scores, labels)
    if bt:
        t_acc, acc_dec, t_j, j, tpr, fpr = bt
        print(f"  melhor threshold (max acuracia da decisao aceitar-se-conf>=t): "
              f"t={t_acc:.3f} -> acuracia {acc_dec:.1%}")
        print(f"  melhor threshold (Youden J): t={t_j:.3f}  J={j:.3f}  "
              f"(TPR={tpr:.2f}, FPR={fpr:.2f})")
        print(f"  => sugestao: confidenceThresholdHigh ~= {t_j:.2f} "
              f"(acima disso, aceitar local; abaixo, escalar)")

    # correlacoes com IRT (documentar que IRT preve mal)
    with_irt = [(float(x["difficulty_score"]), float(x["confianca"]), 1 if x["acertou"] else 0)
                for x in val if x["difficulty_score"] not in ("", None) and x["confianca"] != ""]
    if len(with_irt) >= 3:
        d = [t[0] for t in with_irt]
        cf = [t[1] for t in with_irt]
        ac = [float(t[2]) for t in with_irt]
        print("\n>>> IRT (dificuldade humana) prediz a dificuldade-LLM?")
        rs1 = spearman(d, cf)
        rs2 = spearman(d, ac)
        print(f"  Spearman(difficulty_score, confianca): "
              f"{rs1:.3f}" if rs1 is not None else "  (n/d)")
        print(f"  Spearman(difficulty_score, acerto)   : "
              f"{rs2:.3f}" if rs2 is not None else "  (n/d)")
        print("  (|rho| baixo/perto de 0 confirma o doc 01: IRT != dificuldade para a LLM)")
    print("=================================================")


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def main():
    ap = argparse.ArgumentParser(
        description="Calibra thresholds de confianca por ACERTO do modelo (gabarito).")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"default {DEFAULT_URL}")
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="dataset unificado")
    ap.add_argument("--model-name", dest="model_name", default="",
                    help="rotulo do modelo (ex.: qwen2.5-0.5b); vira coluna `model` no CSV "
                         "e nomeia a saida como resultados_acerto_<rotulo>.csv se --out nao for dado")
    ap.add_argument("--out", default=None,
                    help="CSV de saida por questao. Default: resultados_acerto_local.csv "
                         "(ou resultados_acerto_<model-name>.csv se --model-name for dado)")
    ap.add_argument("--n-predict", type=int, default=4,
                    help="tokens gerados (so precisamos da letra). Default 4")
    ap.add_argument("--n-probs", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 = greedy (recomendado p/ avaliar acerto). Default 0.0")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="timeout (s) por questao. Suba em CPU lenta. Default 300")
    ap.add_argument("--limit", type=int, default=0, help="avalia so as N primeiras (0=todas)")
    ap.add_argument("--areas", default="", help="filtra areas, ex.: LC,CH,CN,MT")
    ap.add_argument("--include-images", action="store_true",
                    help="inclui questoes IU=true (default: so texto puro)")
    ap.add_argument("--require-irt", dest="require_irt", action="store_true", default=True,
                    help="so questoes com difficulty_score (default)")
    ap.add_argument("--no-require-irt", dest="require_irt", action="store_false",
                    help="inclui questoes sem IRT")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
