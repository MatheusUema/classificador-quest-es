# Roadmap da pesquisa — elasticidade de LLMs num tutor educacional

> Mapa curto do arco completo da pesquisa: o que já foi feito, o que está em andamento e o que falta, deixando **rastreável a contribuição integradora** do mestrado. Documentos de detalhe: resultados em `03-resultados-validacao-confianca.md`, protocolo multi-modelo em `04-protocolo-estudo-multimodelo.md`, revisão de literatura em `referencias-confianca-logprobs-mcq.md`.

---

## Objetivo geral

Validar **empiricamente** os conceitos de **elasticidade** dos artigos *AIED Unplugged* (Papers 901 e 1750) num **tutor educacional** que transiciona automaticamente entre três tiers de LLM — **local (celular) → servidor local (LAN) → nuvem** — **conservando desempenho pedagógico e custo** (energia, latência, conectividade, disponibilidade offline). A hipótese operacional central é que a **confiança do próprio modelo** (via *logprobs*) é um sinal útil para decidir **quando escalar** de tier.

## Contribuição integradora (capstone do mestrado)

A síntese da pesquisa é um **app funcional e validado empiricamente sob as limitações reais de hardware** (dispositivos-alvo). Não basta caracterizar modelos e políticas no papel: a completude vem de mostrar o sistema **elástico rodando em hardware modesto**, medindo o trade-off real acurácia × custo × disponibilidade offline. É o que **amarra os dois artigos** — o método de transição (Artigo 1) e a validação end-to-end (Artigo 2) — numa entrega única e defensável.

---

## Fases e entregáveis

### Artigo 1 — Método de transição + caracterização multi-modelo por confiança

- **[FEITO]** Dataset **Maritaca + IRT** (`maritaca_enem_irt.csv`): texto limpo + gabarito + dificuldade IRT, 389 questões de texto puro. Ver doc 03 §2.
- **[FEITO]** **Harness de acurácia/confiança** (`evaluate_local_accuracy.py`): MC A–E, `temperature 0`, confiança = `exp(logprob)` da letra; métricas de AUC/ECE/risco-cobertura e simulação de políticas (`analyze_accuracy.py`).
- **[FEITO]** **Caracterização de 4 modelos locais** (Gemma-3-1B, Qwen2.5-0.5B, Qwen2.5-1.5B, Llama-3.2-1B): Qwen2.5-1.5B domina (acurácia 47,8%, AUC 0,789, ECE 0,064); superconfiança severa é **específica do Gemma**. Ver doc 03 §5.
- **[FEITO]** **Revisão de literatura** sobre confiança via logprobs em MCQ (discriminação × calibração; debiasing; verbalizada/P(True); matemática). Ver `referencias-confianca-logprobs-mcq.md`.
- **[EM ANDAMENTO]** **Experimentos de robustez do sinal**: confiança **verbalizada / P(True)** (`evaluate_verbalized_confidence.py`) e **debiasing de opção** (permutação/PriDe) — testar se batem/superam o logprob (hipótese Tian et al.).
- **[PENDENTE / limitado por hardware]** **Tier servidor** (7–13B via `llama.cpp` na LAN): bake-off de candidatas exige GPU/servidor que hoje não temos (ver riscos).

### Adaptação do app  **[PENDENTE — explícito]**

Ponte entre a pesquisa e o produto; ainda **não iniciada**. Quatro frentes:

1. **Escolher o modelo por tier a partir dos resultados.** Hoje o candidato local recomendado é o **Qwen2.5-1.5B** (domina acurácia *e* qualidade do sinal); o Gemma-3-1B, fixado no código original, é o pior para roteamento por confiança.
2. **Decisão de arquitetura do runtime local.** O **MediaPipe não expõe logprobs** (doc 03 §8), então usar a confiança on-device exige **migrar o tier local para `llama.cpp` (JNI)** ou runtime equivalente que exponha logits. É uma decisão de engenharia, não só de pesquisa.
3. **Integrar a política de roteamento**: threshold **por modelo** escolhido pela **curva risco-cobertura** (não um número global), com **matemática escalada por padrão** (AUC ≈ aleatória em MT em todos os modelos).
4. **Logar área e confiança** em campo, para validação real (e para inferir a área quando o app não a conhece a priori).

### Artigo 2 — Validação end-to-end da elasticidade (síntese do mestrado)

- **[PENDENTE]** Rodar o **app com os três métodos** (sempre-local / elástico / sempre-nuvem) sob **hardware real** (dispositivos-alvo), traçando a **fronteira de Pareto acurácia × custo × latência × disponibilidade offline** contra os baselines.
- **[PENDENTE]** Incluir **modalidade** texto/imagem (proxy `description` e/ou visão), fechando o argumento de que a elasticidade entrega mais valor por recurso que qualquer tier fixo.

---

## Dependências e riscos

- **Hardware.** Notebook fraco e **GPU inutilizável** localmente; o **bake-off do tier servidor** e os testes de escala **dependem de GPU/nuvem** ainda a viabilizar. É o principal gargalo do cronograma.
- **Disponibilidade de logprobs por tier.** Local **MediaPipe = não**; servidor **`llama.cpp` = sim**; nuvem **= só via API crua** (Gemini `responseLogprobs`, formato a confirmar). Isso condiciona onde a confiança é medível de verdade.
- **Lacuna MCQ × perguntas abertas.** A confiança é **medida** em múltipla escolha, mas o app responde perguntas **abertas** em produção; os vieses de MCQ são preocupação de **medição** e a transferência precisa ser validada em formato aberto (ameaça à validade + trabalho futuro).
- **Contaminação de treino** (ENEM público) e **efeito de quantização** (Q4) — mitigações no protocolo (doc 04 §10–11).

---

## Documentos relacionados

- `03-resultados-validacao-confianca.md` — resultados single-model + comparação multi-modelo (§5).
- `04-protocolo-estudo-multimodelo.md` — protocolo de caracterização por tier (local + servidor), métricas e ameaças.
- `referencias-confianca-logprobs-mcq.md` — base bibliográfica (discriminação × calibração, debiasing, verbalizada/P(True), matemática).
