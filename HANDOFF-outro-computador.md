# HANDOFF — rodar os experimentos numa máquina mais rápida

> **Para o agente (Cowork/Claude com acesso a shell e arquivos) da outra máquina.** Este documento é autocontido: leia-o inteiro, depois execute/guie o usuário passo a passo. Objetivo: **rodar mais rápido (idealmente com GPU) o que ainda falta** da pesquisa. Não altere nenhum app; você só roda scripts de avaliação em `data/maritaca/`.
>
> Placeholders a substituir na outra máquina: `<REPO_DIR>` (onde o repositório será clonado/copiado), `<MODELS_DIR>` (pasta dos GGUF), `<LLAMA_DIR>` (pasta do llama.cpp / `llama-server`).

---

## 1. Contexto da pesquisa (por que isto existe)

Pesquisa de **mestrado (USP)** validando empiricamente **elasticidade**: um tutor educacional que **transiciona entre LLM local (celular) → servidor local (LAN) → cloud**, decidindo **quando escalar** de tier pela **confiança do próprio modelo** — definida como a média de `exp(logprob)` do token da resposta escolhida. A pergunta central é se essa confiança **separa acerto de erro** bem o bastante para rotear.

**O que já foi feito e os achados-chave (números reais):**

- **Dataset unificado** `data/maritaca/maritaca_enem_irt.csv`: ENEM 2022–2024 (base Maritaca, texto limpo + gabarito) casado com **dificuldade IRT** do INEP. 540 itens; **531 com IRT**; **389 de texto puro** (`IU=false`) — o conjunto usado nas avaliações.
- **Caracterização de 4 modelos locais** (todos Q4_K_M via `llama.cpp`): **Qwen2.5-1.5B domina** — acurácia **47,8%**, **AUC 0,793**, **ECE 0,064**. **Gemma-3-1B** é **outlier de superconfiança** (ECE **0,41**). **Matemática é o calcanhar universal**: AUC ≈ aleatória (0,46–0,53) em todos.
- **Comparação de sinais de confiança** (mesma resposta): **logprob > verbalizada > P(True)**. No Qwen: logprob AUC 0,793; verbalizada 0,624 (superconfiante); P(True) 0,558. Verbalizada/P(True) são piores ou inviáveis de capturar em modelos pequenos (contraria Tian et al. 2023, que valia para modelos grandes).
- **Debiasing de opção** (permutar a ordem das alternativas): revelou **viés de posição** — o modelo tende a favorecer a letra **"E"** — e uma **AUC "honesta" ~0,70** quando se agrega sobre as permutações. Guarda contra a objeção de Zheng et al. (ICLR 2024).

Detalhes e tabelas em `docs/03-resultados-validacao-confianca.md`; protocolo em `docs/04-protocolo-estudo-multimodelo.md`; arco/roadmap em `docs/00-roadmap-pesquisa.md`.

**Nesta máquina rápida**, o alvo é **acelerar o que falta** (debiasing completo, um ponto de **tier servidor**, e re-rodadas na GPU).

---

## 2. O que copiar para esta máquina

O essencial é a pasta **`data/maritaca/`** inteira — em especial:

- **Scripts:** `evaluate_local_accuracy.py`, `aggregate_multimodel.py`, `evaluate_verbalized_confidence.py`, `evaluate_option_debias.py`, `analyze_accuracy.py`, `build_maritaca_irt.py`.
- **Dataset:** `maritaca_enem_irt.csv` (não precisa reconstruir; só rode `build_maritaca_irt.py` se quiser regenerá-lo).
- **`docs/`** (03, 04, 00-roadmap, referências) para contexto.

Formas de trazer, da mais limpa à mais simples:

```bash
# (A) se o repositório estiver no GitHub — via mais limpa:
git clone <URL_DO_REPO_classificador-questoes> <REPO_DIR>

# (B) senão, copie a pasta do projeto por nuvem/USB para <REPO_DIR>
```

> **Os modelos GGUF NÃO precisam ser copiados** — baixá-los de novo (seção 4) é mais fácil e rápido que transferir arquivos grandes.

---

## 3. O que instalar (assuma a máquina zerada)

**3.1 Python 3** (3.10+). Verifique: `python --version`. Os scripts são **stdlib** para o núcleo; para as figuras do `analyze_accuracy.py`/`aggregate_multimodel.py` instale (opcional):

```bash
pip install numpy matplotlib   # só para os PNGs; as métricas/CSVs funcionam sem eles
```

**3.2 llama.cpp — DETECTE A GPU e escolha a build.** Rode `nvidia-smi`:

- **Tem GPU NVIDIA moderna** (aparece no `nvidia-smi`): baixe a build **CUDA** do llama.cpp (releases de `ggml-org/llama.cpp`, ex.: `llama-<versão>-bin-win-cuda-x64.zip`) **+ o pacote `cudart`** correspondente (DLLs do runtime CUDA, no mesmo release) e descompacte tudo junto em `<LLAMA_DIR>`. É **ordens de magnitude mais rápido**. Ao subir o servidor, use **`--n-gpu-layers 99`** (joga todas as camadas na GPU).
- **Sem GPU utilizável:** baixe a build **CPU** (ex.: `llama-<versão>-bin-win-cpu-x64.zip`, ou a variante AVX2). Sem `--n-gpu-layers`.

> **Aviso importante sobre `--n-probs`:** algumas builds do `llama-server` **não aceitam `--n-probs` como flag de linha de comando** e recusam iniciar. **Você não precisa dessa flag** — os scripts pedem `n_probs` **por requisição** (no JSON). Se o servidor reclamar de `--n-probs`, **remova-a** do comando; os experimentos continuam funcionando.

Teste rápido do servidor depois de instalar: `<LLAMA_DIR>/llama-server --help` deve listar as opções.

---

## 4. Baixar os modelos (para `<MODELS_DIR>`)

Q4_K_M em todos. Use `huggingface-cli` (robusto) **ou** download direto pelo URL `resolve`. Confirme o **nome exato do arquivo** na aba *Files* de cada repositório (nomes mudam entre versões).

```bash
pip install -U "huggingface_hub[cli]"

# TIER LOCAL (os 4 já caracterizados)
huggingface-cli download lmstudio-community/gemma-3-1B-it-GGUF  gemma-3-1B-it-Q4_K_M.gguf          --local-dir <MODELS_DIR>
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF        qwen2.5-0.5b-instruct-q4_k_m.gguf  --local-dir <MODELS_DIR>
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF        qwen2.5-1.5b-instruct-q4_k_m.gguf  --local-dir <MODELS_DIR>
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF   Llama-3.2-1B-Instruct-Q4_K_M.gguf  --local-dir <MODELS_DIR>

# TIER SERVIDOR (aproveitar a máquina rápida) — candidato 7B
huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF     Qwen2.5-7B-Instruct-Q4_K_M.gguf    --local-dir <MODELS_DIR>
```

URLs `resolve` equivalentes (fallback com `curl -L -o`), confirmando o filename no repo:

```
https://huggingface.co/lmstudio-community/gemma-3-1B-it-GGUF/resolve/main/gemma-3-1B-it-Q4_K_M.gguf
https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

> Qwen 0.5B/1.5B GGUF oficiais às vezes vêm com nome ligeiramente diferente (ex.: sufixo de shard). Se o download falhar, abra a aba *Files* do repo e copie o nome real.

---

## 5. Como rodar cada experimento

**Fluxo geral — um modelo por vez:** sobe o `llama-server` com o GGUF → roda o eval (na mesma porta 8080) → encerra o servidor → próximo modelo. Todos os scripts ficam em `<REPO_DIR>/data/maritaca/` (rode `cd` para lá antes).

```bash
cd <REPO_DIR>/data/maritaca
```

**5.1 Subir o servidor** (uma janela/terminal; deixe rodando). Com GPU:

```bash
<LLAMA_DIR>/llama-server -m <MODELS_DIR>/qwen2.5-1.5b-instruct-q4_k_m.gguf --port 8080 --n-gpu-layers 99
# sem GPU: remova --n-gpu-layers 99
# se a build reclamar de --n-probs, simplesmente não passe essa flag (ver seção 3.2)
```

**5.2 Acurácia + confiança logprob (por modelo)** — gera `resultados_acerto_<rótulo>.csv`:

```bash
python evaluate_local_accuracy.py --url http://127.0.0.1:8080 --model-name qwen2.5-1.5b
# repita para cada modelo, trocando o GGUF no servidor e o --model-name:
#   gemma-3-1b | qwen2.5-0.5b | qwen2.5-1.5b | llama-3.2-1b
```

**5.3 Consolidar os modelos** (offline; depois de rodar vários) — gera `comparativo_modelos.csv` e (se houver matplotlib) `analise/fig_cmp_*.png`:

```bash
python aggregate_multimodel.py
```

**5.4 Comparação de sinais logprob vs verbalizada vs P(True)** — gera `verbalized_vs_logprob_<rótulo>.csv`. Faz 3 chamadas por questão (mais lento); o servidor deve expor `n_probs` alto (o script já pede `--n-probs-ptrue 10` por requisição):

```bash
python evaluate_verbalized_confidence.py --url http://127.0.0.1:8080 --model-name qwen2.5-1.5b --limit 40   # sanity
python evaluate_verbalized_confidence.py --url http://127.0.0.1:8080 --model-name qwen2.5-1.5b              # completo
```

**5.5 Debiasing / robustez de opção** (5 permutações por questão; caro) — gera `resultados_debias_<rótulo>.csv` e `debias_resumo_<rótulo>.csv`:

```bash
python evaluate_option_debias.py --url http://127.0.0.1:8080 --model-name qwen2.5-1.5b --limit 40   # sanity (200 chamadas)
python evaluate_option_debias.py --url http://127.0.0.1:8080 --model-name qwen2.5-1.5b              # completo (~1945 chamadas)
```

> Dica: comece **sempre** pelo `--limit 40` para validar que o servidor responde e os números fazem sentido; só então rode o conjunto completo.

---

## 6. Tarefas pendentes prioritárias (nesta ordem)

1. **Debiasing de opção completo no Qwen2.5-1.5B** (§5.5, sem `--limit`). **Pode já estar rodando na máquina lenta — confirme com o usuário** antes de duplicar; se estiver, pule para a tarefa 2.
2. **Um ponto de tier servidor: Qwen2.5-7B-Instruct.** Suba o 7B (com `--n-gpu-layers 99`) e rode `evaluate_local_accuracy.py --model-name qwen2.5-7b`; depois `aggregate_multimodel.py`. Objetivo: obter o **gradiente local → servidor** (acurácia e AUC do 7B ao lado dos 4 locais).
3. **(Opcional) Re-rodar os 4 modelos locais + verbalizada na GPU**, muito mais rápido, para consolidar/estabilizar os números (útil porque estimativas em amostra pequena são instáveis — ex.: P(True) foi 0,749 em 40 e 0,558 em 389).

---

## 7. Como devolver os resultados

Traga de volta para a máquina principal (ou **cole os sumários impressos no chat**) os CSVs gerados, para consolidar em `docs/03-resultados-validacao-confianca.md`:

- `resultados_acerto_*.csv` (um por modelo)
- `comparativo_modelos.csv`
- `verbalized_vs_logprob_*.csv`
- `resultados_debias_*.csv` e `debias_resumo_*.csv`
- (se gerou) os PNGs em `data/maritaca/analise/`

Além dos arquivos, copie do console os blocos de **SUMARIO** que cada script imprime — eles já trazem acurácia por área, AUC, ECE e thresholds prontos para colar.
