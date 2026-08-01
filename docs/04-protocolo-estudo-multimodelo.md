# Protocolo do estudo multi-modelo: caracterização de LLMs candidatas por tier (local + servidor)

> Plano de pesquisa (acionável) para avaliar **várias LLMs candidatas por tier** — **local**
> e **servidor local** — quanto a competência e, sobretudo, à **qualidade do sinal de
> confiança** que fundamenta o método de transição entre tiers. Estende o Artigo 1, cujos
> resultados de modelo único estão em `03-resultados-validacao-confianca.md`.
>
> **Escopo desta rodada:** apenas **local + servidor** (ambos via `llama.cpp`). O tier
> **nuvem fica adiado** para o Artigo 2 (ver §2.1). Os refinamentos metodológicos vêm da
> revisão de literatura em `referencias-confianca-logprobs-mcq.md`.
>
> **Este é um rascunho para você revisar.** Os pontos marcados **[CONFIRMAR]** dependem da
> sua decisão antes de rodar.

---

## 1. Objetivo e perguntas de pesquisa

O estudo de modelo único (doc 03) mostrou que, para o Gemma-3-1B, a confiança é um sinal
**fraco** (AUC 0,66) e **mal calibrado** (ECE 0,406). Mas isso é uma propriedade **daquele
modelo**, não necessariamente do método: outro modelo pode ter confiança mais discriminativa.
Como o método de transição depende inteiramente da qualidade desse sinal, precisamos
**caracterizá-lo em várias LLMs** por tier e usar isso para escolher o modelo de cada tier.

Uma distinção conceitual que organiza todo o estudo (e vem da literatura, §Referências): para
**roteamento**, o que importa é a **discriminação** — a capacidade de *ordenar* acertos acima
de erros (ROC AUC, curva risco-cobertura) — e **não** a calibração (ECE). Calibração ruim
significa apenas que o número não pode ser lido como probabilidade literal; ela é **corrigível
a posteriori** (temperature scaling, Guo et al., 2017) e **não impede** o roteamento.
Discriminação, ao contrário, não pode ser "criada" onde não existe. Por isso a métrica-chave
é AUC/risco-cobertura, com ECE como métrica secundária (diagnóstico de superconfiança).

Perguntas de pesquisa:

- **PP1 (competência).** Qual a acurácia de cada candidata no ENEM, global e por área?
- **PP2 (sinal de confiança) — central.** Quão bem a confiança de cada candidata **discrimina**
  acerto de erro (**AUC**, **curva risco-cobertura**)? E, secundariamente, quão calibrada é
  (**ECE**, antes e depois de temperature scaling)?
- **PP2b (sinais alternativos).** A **confiança verbalizada** / `P(True)` discrimina melhor que
  os logprobs, por modelo? (comparação de AUC entre sinais)
- **PP3 (custo).** Qual o custo de cada candidata no seu tier — **on-device** (latência,
  tokens/s, RAM, energia) para local, e **latência de LAN** + throughput para o servidor?
- **PP4 (decisão).** Combinando PP1–PP3, qual modelo escolher para cada tier, e sob quais
  premissas de orçamento?

O produto é, para cada tier, uma **fronteira de Pareto** (competência × custo) anotada com a
**qualidade do sinal de confiança** — a base objetiva para a escolha por tier.

---

## 2. Substrato de medição: llama.cpp nos dois tiers

**Decisão de medição (não de produção):** rodar **todas** as candidatas via `llama.cpp` —
local e servidor. Justificativa metodológica:

- **Logprobs uniformes e comparáveis.** A discriminação/calibração (PP2) exige *logprobs* por
  token. O `llama.cpp` os expõe de forma idêntica para qualquer modelo GGUF, tornando
  AUC/ECE/risco-cobertura **comparáveis entre candidatas** — o requisito central do estudo.
- **MediaPipe (tier local de produção) não expõe logprobs.** `generateResponse()` devolve só
  texto (ver doc 03, §7). Medir a confiança do tier local *no runtime de produção* é
  impossível hoje.

**Consequência a declarar sempre:** a caracterização usa um **runtime de medição** (`llama.cpp`)
que **difere do runtime de produção do tier local** (MediaPipe). Isso é válido para *comparar
candidatas entre si* sob condições idênticas, mas números absolutos de confiança podem não se
reproduzir em produção — ameaça tratada na §11. Candidatos locais rodam **on-device** (build
Android/ARM do `llama.cpp`); candidatos de servidor rodam na **máquina de servidor** (LAN).

### 2.1 Por que a nuvem fica fora desta rodada (e volta no Artigo 2)

O tier **nuvem** está **fora do escopo deste estudo**, por duas razões: (i) **incerteza sobre
logprobs** — o SDK do Firebase AI Logic não os expõe, e medir confiança da nuvem exigiria a
API crua do Gemini (`responseLogprobs`), cuja disponibilidade/formato ainda estamos
verificando, além de ser um runtime distinto do `llama.cpp` (quebra a comparabilidade); e
(ii) **foco** — caracterizar bem local+servidor com sinal comparável é o passo que fundamenta
o método de transição. A nuvem **permanece no roadmap**: entra na validação end-to-end do
**Artigo 2** (§10), onde a confiança da nuvem pode ser opcional (a nuvem é o tier "forte" de
destino, não necessariamente precisa expor confiança para o roteamento).

---

## 3. Candidatos por tier

Quantização **Q4_K_M** (GGUF) como padrão, para refletir o uso real em dispositivos modestos;
opcionalmente comparar Q4 vs Q8 num modelo, para medir o efeito da quantização (§11).

### 3.1 Tier local (on-device) — **FIXADO**

Lista **confirmada**. Todos *instruct*, quantização **Q4_K_M** GGUF, medidos via `llama.cpp`.
O critério de escolha serve à **PP2**: cobrir a faixa **0,5B–1,5B** em **três famílias distintas**
(Gemma, Qwen, Llama) para verificar se a **qualidade do sinal de confiança** (AUC/ECE) varia por
**família e por tamanho** — não basta um único modelo pequeno para concluir sobre o método.

| Modelo (instruct, Q4_K_M) | Tamanho aprox. em disco (Q4_K_M) | Fonte GGUF sugerida (HuggingFace) | Observação |
|---|---|---|---|
| **Gemma-3-1B-it** | ~0,8 GB | `ggml-org` / `unsloth` (GGUF oficiais Gemma-3) | **Baseline** — já avaliado nos resultados do doc 03 |
| **Qwen2.5-0.5B-Instruct** | ~0,4 GB | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (oficial) | Menor da faixa; piso de custo |
| **Qwen2.5-1.5B-Instruct** | ~1,0 GB | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` (oficial) | Mesma família do 0,5B → isola efeito de **tamanho** |
| **Llama-3.2-1B-Instruct** | ~0,8 GB | `bartowski` / `unsloth` (GGUF de Llama-3.2-1B) | Terceira família no mesmo ~1B → isola efeito de **família** |

*(Tamanhos são aproximados e variam com a versão do GGUF; confirmar o byte-count real do arquivo
baixado e registrá-lo na saída, conforme §7.)*

**Extensão opcional — "teto local" (~3B):** **Qwen2.5-3B-Instruct** *ou* **Llama-3.2-3B-Instruct**
em Q4_K_M (~2 GB). Como 3B em Q4 (~2 GB) **pesa em celulares** (RAM/carga do modelo em aparelhos
modestos), é **extensão, não obrigatório** — entra só se os dispositivos-alvo (§8) comportarem.

### 3.2 Tier servidor (LAN) — **[CONFIRMAR/EDITAR a lista]**

Sugestão inicial (~7–13B, via `llama-server` na LAN): Qwen2.5-7B-Instruct; Llama-3.1-8B-Instruct;
Gemma-2-9B-IT. **O usuário decide depois** quais candidatas entram, tamanhos e quantizações.

> Modelos **com visão** para o subconjunto multimodal são tratados na §5 (extensão).

---

## 4. Conjunto de avaliação (texto puro)

- **Base:** `data/maritaca/maritaca_enem_irt.csv` — **389 questões de texto puro** (`IU=false`)
  com gabarito e dificuldade IRT, 2022–2024.
- **Mesmo protocolo do doc 03:** múltipla escolha A–E, "responda só a letra",
  **`temperature = 0`** (greedy), extração da primeira letra A–E, confiança = `exp(logprob)`
  do token da letra.
- **Rótulo:** acerto vs gabarito (não IRT). O IRT entra apenas como covariável de análise.

O subconjunto multimodal (questões com imagem) é avaliado à parte — §5.

---

## 5. Subconjunto multimodal (questões com imagem, `IU=true`)

**Decisão: incluir.** As **142 questões `IU=true`** do conjunto de ouro têm imagem e, na base
Maritaca, também uma **`description`** — o texto acessível do caderno "ledor" que descreve a
figura para pessoas com deficiência visual. Como a maioria das candidatas via `llama.cpp` é
**text-only**, propomos duas formas de avaliar, uma imediata e uma como extensão:

- **(a) Proxy textual via `description` — caminho simples e imediato.** Injetar a `description`
  no lugar da imagem e medir se o modelo pequeno resolve a questão a partir da descrição. É o
  caminho recomendado para começar: reusa o harness text-only sem mudanças, isola o efeito
  "descrição recupera a competência perdida?" e não exige modelos de visão. Marca-se cada
  questão como `multimodal_proxy=true` na saída, para reportar separadamente.
- **(b) Modelos com visão via `llama.cpp` multimodal — extensão.** Rodar candidatas
  *vision-capable* (ex.: Qwen2-VL, Gemma-3 vision) pelo suporte multimodal do `llama.cpp`,
  passando a imagem real. Fornece o número "honesto" de multimodalidade, mas exige build/pesos
  de visão e é mais custoso; fica como extensão. **[CONFIRMAR se/quando (b) entra.]**

> **[DECISÃO]** Começar por (a) para todas as candidatas text-only; considerar (b) apenas para
> um ou dois modelos de visão como extensão. Confirme.

---

## 6. Reuso e adaptação do harness

Reaproveitar `evaluate_local_accuracy.py` (doc 03), tornando-o **multi-modelo** e incorporando
os passos metodológicos da literatura, com mudanças mínimas:

- **`--model <id>` e `--url <endpoint>` por candidata** (cada modelo sobe num `llama-server`
  próprio, ou troca-se o `-m` entre execuções). Registrar `model_id`, `tier` e as métricas de
  custo (§7) em cada linha do CSV.
- **Debiasing de opção (novo passo — Zheng et al., 2024).** Antes de tratar `exp(logprob)` da
  letra como confiança, mitigar o **viés de seleção de MCQ** (modelos têm preferência posicional
  por certas letras). Duas opções, da mais simples à mais completa: (i) **permutar a ordem das
  alternativas** k vezes por questão e marginalizar/promediar a confiança sobre as permutações;
  (ii) **PriDe** (estimar o *prior* de posição do modelo em um subconjunto e removê-lo). Sem
  isso, parte do "sinal" de confiança é viés posicional, não crença. Reportar a **acurácia e a
  AUC com e sem debiasing**, para quantificar o viés por modelo. **[CONFIRMAR k de permutações,
  ex.: k=4]**
- **Probabilidade da letra, não da sequência inteira (Holtzman et al., 2021).** Manter a
  confiança sobre o **token da letra** (já é o caso), evitando *surface form competition* /
  *length bias* de comparar textos completos de alternativas.
- **Sinal alternativo — confiança verbalizada / `P(True)` (novo — Tian et al., 2023; Kadavath
  et al., 2022).** Em uma segunda passada (ou prompt adicional), coletar: (i) a **confiança
  verbalizada** (pedir ao modelo "de 0 a 100%, quão certo você está?") e/ou (ii) **`P(True)`**
  (probabilidade que o modelo atribui a "a resposta escolhida está correta?"). Comparar a **AUC
  do logprob-confidence vs verbalizado** por modelo — em modelos alinhados/RLHF a verbalizada
  às vezes discrimina melhor.
- **Manter idêntico** o restante do protocolo (prompt base, extração da letra, `temperature=0`),
  para que as diferenças venham do **modelo**, não do harness.
- **Agregador** (estender `analyze_accuracy.py`) que consolida os `resultados_*_<model>.csv` e
  produz as tabelas/figuras comparativas da §9.

Saída sugerida: `data/maritaca/multimodelo/resultados_<tier>_<model>.csv` e um
`sumario_multimodelo.csv` (uma linha por modelo × sinal).

---

## 7. Métricas por modelo/tier

Para cada candidata:

**Competência (PP1).** Acurácia global e por área (LC/CH/CN/MT); N e intervalo de confiança
(§9). Reportar **com e sem debiasing de opção**.

**Qualidade do sinal de confiança (PP2) — o coração do estudo.**
- **Discriminação (primária):** **ROC AUC** (confiança prevendo acerto), global e por área; e a
  **curva risco-cobertura** (*accuracy vs coverage*) — para cada cobertura (fração de questões
  mantidas localmente), qual a acurácia das mantidas. É a curva que o roteamento realmente usa:
  o limiar é escolhido nela, para um **orçamento-alvo** de cobertura/escalonamento (não por
  Youden global).
- **Calibração (secundária):** **ECE** antes e **depois de temperature scaling** (Guo et al.,
  2017), ajustado num split de validação. Documenta a superconfiança e quanto dela é corrigível.
- **Sinais comparados (PP2b):** AUC do **logprob-confidence** vs **verbalizado/`P(True)`** por
  modelo; opcional um ensemble simples dos dois.
- **Incerteza token-level (opcional):** entropia/quantis da distribuição de tokens, não só a
  média de `exp(logprob)` — costuma dar melhor trade-off custo-qualidade em cascatas.

**Tratamento especial de Matemática/raciocínio.** No doc 03 a AUC em MT foi ~0,53 (nula) — um
padrão previsto pela literatura (CoT descalibra a probabilidade do token final). Para itens de
MT/raciocínio, prever no protocolo: **(i) escalonar por padrão** (tratar MT como "sempre difícil
para o modelo pequeno") **ou (ii) self-consistency** (votar sobre múltiplas amostragens com
`temperature>0`) como sinal de confiança alternativo. Reportar AUC de MT sob cada tratamento.

**Custo (PP3).**
- **Local (on-device):** tempo por questão, **tokens/s** (ingestão do prompt e geração
  separadas — enunciados do ENEM são longos e a ingestão domina), **RAM** em uso e pico,
  **energia** por questão (proxy; §8), **tamanho do modelo** em disco.
- **Servidor (LAN):** latência total incluindo rede, throughput, e RAM/VRAM do servidor.

---

## 8. Protocolo de medição on-device

Para os candidatos locais (e a coleta de custo que só existe no device):

- **Dispositivos-alvo [CONFIRMAR — o usuário fornecerá].** Não presumidos aqui. Ao definir,
  registrar modelo do aparelho, SoC, RAM e versão do Android.
- **Build:** `llama.cpp` compilado para **Android/ARM** (`arm64-v8a`), com as mesmas flags entre
  candidatas; fixar nº de threads e desabilitar governadores agressivos se possível.
- **Warm-up:** descartar as 1–2 primeiras questões (carga de modelo, JIT, caches) antes de medir.
- **Repetições:** cada questão rodada **k vezes [CONFIRMAR, ex.: k=3]** para custo (latência/
  energia variam), reportando mediana e dispersão; acurácia/confiança com `temperature=0` são
  determinísticas (1 vez basta) — exceto onde usarmos self-consistency (MT), que exige múltiplas
  amostragens por desenho.
- **Controle térmico:** intervalos entre execuções e monitoramento de temperatura;
  descartar/repetir sob *thermal throttling*; alternar a ordem das candidatas para não penalizar
  sempre a mesma.
- **Latência:** medir separadamente **ingestão do prompt** e **geração**.
- **Energia:** via `BatteryManager`/estatísticas do device como **proxy** (não wattímetro); medir
  carga consumida por bloco de N questões com tela/rede controladas.
- **RAM:** RSS/PSS do processo, pico incluído.

**Ressalvas de medição:** energia é proxy; latência varia com temperatura/bateria; o build de
medição não é o app de produção. Registrar tudo junto dos números.

---

## 9. Análise

- **Fronteiras de Pareto por tier:** acurácia × custo (uma fronteira para latência, uma para
  energia, uma para RAM no local; latência/throughput no servidor), marcando as candidatas
  não-dominadas. Responde PP4. (Enquadramento de cascata custo-qualidade: FrugalGPT, Chen et
  al., 2023.)
- **Comparação da discriminação entre modelos (central):** AUC e **curvas risco-cobertura** de
  todas as candidatas lado a lado — mostra se *algum* modelo torna o roteamento por confiança
  realmente viável, e para cada um qual acurácia local é sustentável a cada nível de cobertura.
- **Seleção de limiar por orçamento:** para o(s) modelo(s) escolhido(s), extrair da curva
  risco-cobertura o limiar que atinge o **orçamento-alvo de escalonamento** — **não** Youden
  global (que ignora o custo). Reportar o limiar recomendado por orçamento.
- **Efeito da recalibração:** ECE antes/depois de temperature scaling, e se a recalibração muda
  o limiar recomendado (a discriminação/AUC não muda com scaling monotônico, mas a leitura do
  limiar sim).
- **logprob vs verbalizado (PP2b):** qual sinal discrimina melhor por modelo; se um ensemble
  simples ganha.
- **Debiasing de opção:** de quanto o viés posicional inflava acurácia/AUC (com vs sem).
- **Matemática:** AUC de MT sob escalonamento-padrão vs self-consistency.
- **Simulação de políticas por modelo:** reusar as políticas do doc 03 (sempre-local,
  sempre-escalar, limiar global, área-consciente, pré-filtro IRT) para o modelo escolhido de
  cada tier, checando se "área-consciente vence" se mantém.
- **Incerteza:** N por célula (área × modelo — ~77–117 por área, menos em recortes) e intervalos
  de confiança (ex.: Wilson para proporções); não sobre-interpretar diferenças de 1–2 pontos.

---

## 10. Ligação com os artigos

- **Artigo 1 — Método de transição + caracterização multi-modelo (local + servidor).** Formaliza
  o método de roteamento (discriminação da confiança + regra por área + tratamento de MT) e o
  caracteriza sobre várias LLMs nos tiers local e servidor (este protocolo), respondendo "quando
  escalar" e "com qual modelo em cada tier".
- **Artigo 2 — Validação end-to-end da elasticidade (inclui a nuvem).** Mede o sistema completo
  (com os modelos escolhidos) em cenários **offline / LAN / internet**, **reintroduzindo o tier
  nuvem**, traçando a fronteira de Pareto **acurácia × custo × disponibilidade offline** contra
  baselines (**sempre-local**, **sempre-nuvem**, **limiar global**) e incorporando **modalidade**
  (texto vs imagem via `description` e/ou visão). Fecha o argumento dos papers AIED de que a
  elasticidade entrega mais valor por recurso do que qualquer tier fixo.

---

## 11. Ameaças à validade

- **Confiança medida em MCQ, mas produção é aberta (conceitual — importante).** Aqui medimos a
  confiança em **múltipla escolha** (probabilidade da letra), mas em produção o app responde
  **perguntas abertas** do aluno, onde não há "letra" e os vieses de MCQ (seleção/posição,
  *surface form*) não se aplicam da mesma forma. Portanto, os vieses de MCQ são preocupação de
  **medição** — mitigados por debiasing (§6) — e a transferência da confiança para o formato
  aberto **precisa ser validada à parte** (ver trabalho futuro no Artigo 2: repetir a análise de
  discriminação em um conjunto de perguntas abertas com avaliação de correção).
- **Contaminação de treino.** O ENEM é público e pode estar no treino das candidatas — de forma
  **desigual entre modelos**, distorcendo a comparação. Mitigar reportando por ano (edições
  recentes = menos contaminação provável) e sendo cauteloso com números absolutos.
- **Efeito da quantização.** Q4 degrada o modelo, possivelmente de forma desigual entre famílias.
  Mitigar comparando Q4 vs Q8 em ao menos um modelo.
- **Descalibração em raciocínio (CoT/Matemática).** Prevista pela literatura; endereçada pelo
  tratamento especial de MT (§7), mas limita o alcance da confiança nessa área.
- **Energia como proxy.** Medida por API do device, não por instrumento; comparações relativas,
  não absolutas.
- **Área como verdade-fundamental.** As políticas por área usam a área conhecida no dataset; em
  produção ela precisaria ser inferida (UI/classificador) — o estudo mede o **teto** dessas
  políticas.
- **Proxy `llama.cpp` ≠ runtime de produção do tier local (MediaPipe).** Válido para comparar
  candidatas sob condições idênticas, não para prever a confiança exata em produção.
- **Sensibilidade ao prompt.** Um único template pode favorecer certas famílias; opcionalmente
  testar 2 variantes de prompt para checar robustez.

---

## 12. Checklist para começar

1. **[CONFIRMAR]** lista final de candidatas local e servidor (§3) e quantizações (Q4 padrão;
   Q4 vs Q8 em um modelo?).
2. **[CONFIRMAR]** dispositivos-alvo on-device e **k** de repetições de custo (§8).
3. **[CONFIRMAR]** subconjunto multimodal: começar por (a) proxy `description` para todas; (b)
   modelos de visão como extensão? (§5)
4. **[CONFIRMAR]** k de permutações para o debiasing de opção (§6).
5. Adaptar `evaluate_local_accuracy.py` para multi-modelo + debiasing + coleta de confiança
   verbalizada/`P(True)` (§6).
6. Rodar por candidata (texto puro e proxy multimodal); consolidar em `sumario_multimodelo.csv`;
   gerar Pareto, curvas risco-cobertura e a comparação de AUC entre sinais (§9).

---

## Referências

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural
  Networks.* ICML 2017.
- Holtzman, A., West, P., Shwartz, V., Choi, Y., & Zettlemoyer, L. (2021). *Surface Form
  Competition: Why the Highest Probability Answer Isn't Always Right.* EMNLP 2021.
- Kadavath, S., Conerly, T., Askell, A., et al. (2022). *Language Models (Mostly) Know What They
  Know.* Anthropic (arXiv:2207.05221).
- OpenAI (2023). *GPT-4 Technical Report.* arXiv:2303.08774.
- Tian, K., Mitchell, E., Zhou, A., et al. (2023). *Just Ask for Calibration: Strategies for
  Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback.*
  EMNLP 2023.
- Chen, L., Zaharia, M., & Zou, J. (2023). *FrugalGPT: How to Use Large Language Models While
  Reducing Cost and Improving Performance.* arXiv:2305.05176 (TMLR 2024).
- Zheng, C., Zhou, H., Meng, F., Zhou, J., & Huang, M. (2024). *Large Language Models Are Not
  Robust Multiple Choice Selectors.* ICLR 2024 (Spotlight).

> Bibliografia estendida e as evidências a favor/contra (incl. self-consistency, entropia
> semântica, cascatas token-level) estão em `referencias-confianca-logprobs-mcq.md`.
