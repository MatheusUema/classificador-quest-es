# Guia de consolidação — máquina do tier SERVIDOR (ex.: Qwen2.5-7B)

Guia curto e acionável para o agente Claude do **outro computador** (o que rodou o
tier servidor) consolidar os resultados no repositório compartilhado.

> Repo: `MatheusUema/classificador-quest-es` (GitHub). Todos os scripts ficam em
> `data/maritaca/`. Rode os comandos a partir dessa pasta.
> **Respeite o `.gitignore`**: não versione microdados nem `.gguf`.

---

## 1. Contexto

- O **tier LOCAL** já está neste repo (GitHub), consolidado e commitado:
  - 4 modelos: `gemma-3-1b`, `qwen2.5-0.5b`, `qwen2.5-1.5b`, `llama-3.2-1b`.
  - Acurácia (acerto por questão), confiança **verbalizada** vs logprob, e **debiasing**
    de opção (viés de posição) — o pipeline completo do Qwen2.5-1.5B.
- O **tier SERVIDOR** rodou **nesta máquina** (ex.: `Qwen2.5-7B`). Os CSVs desse tier
  existem **só localmente aqui** — ainda não estão no GitHub. O objetivo deste guia é
  trazê-los para o repo e gerar o comparativo local→servidor.

---

## 2. Passos de consolidação

### (a) Sincronizar o repo
```bash
git pull
```

### (b) Colocar os CSVs do 7B em `data/maritaca/`
Garanta que os arquivos gerados aqui estejam na pasta, com o rótulo do modelo no nome
(mesma convenção `<label>` do tier local, ex.: `qwen2.5-7b`):

- `resultados_acerto_<label>.csv`  ← **obrigatório** (saída do `evaluate_local_accuracy.py --model-name <label>`)
- `verbalized_vs_logprob_<label>.csv`  ← se rodou a verbalizada
- `resultados_debias_<label>.csv`  ← se rodou o debiasing de opção

> Confira os nomes: o agregador faz glob `resultados_acerto_*.csv` e usa a coluna
> `model` do CSV como rótulo (cai para o nome do arquivo se estiver vazia).

### (c) Gerar o comparativo INCLUINDO o 7B
```bash
python aggregate_multimodel.py
```
Isso lê **todos** os `resultados_acerto_*.csv` (os 4 locais já versionados + o 7B) e
produz `comparativo_modelos.csv` — uma linha por modelo × métrica: **acurácia** (global
e por área LC/CH/CN/MT), **AUC** (global e por área) e **ECE** (calibração). É o
**gradiente local→servidor**. Se `matplotlib` estiver disponível, também gera
`analise/fig_cmp_acc_area.png` e `analise/fig_cmp_auc_global.png`.

### (d) (Opcional) Paridade com o Qwen2.5-1.5B
Para deixar o 7B com o mesmo conjunto de análises do 1,5B:
```bash
python analyze_accuracy.py                # detalhamento de acurácia/AUC/ECE do 7B
python evaluate_verbalized_confidence.py  # verbalizada vs logprob
python evaluate_option_debias.py          # viés de posição / debiasing por voto
```

### (e) Versionar e sincronizar
```bash
git add data/maritaca/resultados_acerto_<label>.csv \
        data/maritaca/verbalized_vs_logprob_<label>.csv \
        data/maritaca/resultados_debias_<label>.csv \
        data/maritaca/comparativo_modelos.csv \
        data/maritaca/analise/*.png
git commit -m "Adiciona tier servidor (<label>) + comparativo multimodelo"
git push
```
> Não faça `git add` de microdados ou `.gguf` — o `.gitignore` já cobre isso; confira
> com `git status` antes do commit.

---

## 3. O que a consolidação habilita (próximos passos)

- **Servidor (7B) vs local (1,5B)**: quantificar o ganho de **acurácia** e de
  **qualidade de sinal** (AUC / ECE) ao subir de tier — não só acerta mais, mas a
  confiança fica mais discriminativa e melhor calibrada?
- Alimenta a **decisão de modelo por tier** no [doc 04](04-protocolo-estudo-multimodelo.md)
  e a **política de roteamento por confiança** (ver [doc 03](03-resultados-validacao-confianca.md)).
- Fica **pendente**: validade em **formato aberto (não-MCQ)** — o sinal medido aqui é
  todo em múltipla escolha; ver [00-roadmap](00-roadmap-pesquisa.md).

---

## 4. Cuidados

- **`--n-probs`**: algumas builds de `llama-server` no servidor **não aceitam** a flag
  na linha de comando — nesses casos o número de logprobs é pedido **por requisição**
  pelo próprio script. Se subir sem a flag, confirme que a avaliação ainda recebe os
  logprobs esperados.
- **Nomes de arquivo**: confira o `<label>` em todos os CSVs (`resultados_acerto_`,
  `verbalized_vs_logprob_`, `resultados_debias_`). O agregador depende do padrão do
  nome + coluna `model`; rótulo errado vira modelo fantasma no comparativo.
- Rode o `aggregate_multimodel.py` **só depois** de todos os CSVs de acerto estarem na
  pasta — ele agrega o que encontrar no momento da execução.
