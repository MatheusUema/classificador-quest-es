# Confiança de LLM via logprobs em questões de múltipla escolha: revisão de literatura

**Pergunta orientadora:** faz sentido, academicamente, medir a *confiança* de um LLM através de logprobs (probabilidades de token) no contexto de questões de múltipla escolha (MCQ), e usar esse sinal para roteamento elástico entre tiers (local → servidor → cloud)?

**Contexto do projeto:** tutor educacional com roteamento elástico; sinal de confiança = média de `exp(logprob)` do token escolhido; usado para decidir quando escalar de tier. Achados empíricos preliminares em Gemma-3-1B sobre ENEM: discriminação acerto/erro fraca-moderada (**ROC AUC ≈ 0,66**), forte superconfiança (**ECE ≈ 0,40**), sinal quase nulo em matemática.

*Documento produzido em 19/07/2026. Cada fonte está marcada como **peer-reviewed** ou **preprint / relatório técnico**, com venue e link verificável. Ver a seção final de verificação de credibilidade.*

---

## (a) Resumo do veredito

**Sim, é academicamente defensável — mas de forma condicional e com ressalvas fortes.** A literatura sustenta que a probabilidade de token de um LLM carrega sinal real de correção, e que esse sinal é o mecanismo padrão em sistemas de cascata/roteamento que funcionam na prática (FrugalGPT, cascatas do Google). Porém a mesma literatura documenta que esse sinal é frequentemente **mal calibrado** (superconfiança), **enviesado pela forma da MCQ** (viés de posição/token, competição de forma de superfície) e **degradado por alinhamento (RLHF), por tarefas de raciocínio (matemática) e — de forma relevante para você — em modelos pequenos.**

A conclusão operacional é: usar logprobs como confiança é legítimo **se** (1) você medir a probabilidade da *letra/opção* corretamente (não a sequência inteira sem normalizar), (2) tratar vieses específicos de MCQ, (3) recalibrar (p.ex. temperature scaling) antes de confiar em limiares, e (4) reconhecer que discriminação (AUC) — não calibração (ECE) — é o que importa para **decidir escalar de tier**. Para roteamento você precisa que confiança *ordene* acertos acima de erros; não precisa que os números batam com a probabilidade real. Isso muda como interpretar seus próprios resultados (ver seção **e**).

Seus números (AUC ≈ 0,66, ECE ≈ 0,40, matemática ≈ nula) são **exatamente o que a literatura prevê** para um modelo pequeno, instruction-tuned, sem recalibração, em MCQ com componente de raciocínio. Não invalidam a abordagem; indicam que você está no regime "sinal fraco, mal calibrado" que a literatura mapeia — e que admite mitigações conhecidas.

---

## (b) Evidências A FAVOR

**Modelos (grandes) são bem calibrados em MCQ no formato certo — Kadavath et al. (2022).** O trabalho da Anthropic mostra que modelos *grandes* são bem calibrados em conjuntos diversos de questões de múltipla escolha e verdadeiro/falso *quando apresentadas no formato adequado*, e introduz auto-avaliação via `P(True)` e `P(IK)` ("I know"). É a referência canônica de que "modelos sabem, em grande parte, o que sabem". Ressalva embutida no próprio título e resultados: o "(Mostly)" e a queda de calibração de `P(IK)` em tarefas novas. *Preprint / relatório técnico (Anthropic), arXiv 2207.05221.*

**GPT-4 é altamente calibrado em MMLU antes do pós-treino — GPT-4 Technical Report (OpenAI, 2023).** O relatório mostra o modelo *pré-treinado* muito bem calibrado em um subconjunto do MMLU (múltipla escolha): a confiança (logprob) na opção acompanha de perto a probabilidade real de acerto, próxima da diagonal de calibração perfeita. É a evidência mais forte de que logprobs em MCQ *podem* ser um sinal de confiança quase ideal. *Relatório técnico (não peer-reviewed), arXiv 2303.08774.*

**Cascatas/roteamento por confiança de probabilidade funcionam na prática — FrugalGPT (Chen, Zaharia & Zou, 2023/2024).** A cascata de LLMs escala para modelos mais capazes só quando a confiança do modelo mais fraco fica abaixo de um limiar, e reporta igualar o desempenho do melhor LLM individual (GPT-4) com **até ~98% de redução de custo**. É prova de conceito direta de que confiança baseada em probabilidade é usável para exatamente a sua decisão de escalonamento. *Preprint publicado depois em TMLR 2024 (peer-reviewed), arXiv 2305.05176.*

**Incerteza em nível de token melhora cascatas — "Language Model Cascades: Token-level uncertainty and beyond" (Google Research, 2024).** Mostra que incorporar incerteza *token-level* (quantis da distribuição) melhora significativamente o trade-off custo-qualidade das cascatas, superando a simples probabilidade de sequência. Confirma que o sinal de logprob é aproveitável para deferir — desde que refinado. *Preprint / publicação Google Research, arXiv 2404.10136.*

**Temperature scaling recupera calibração de forma barata — Guo et al. (2017).** Estabelece que redes neurais modernas são sistematicamente superconfiantes e que *temperature scaling* (um único parâmetro) recalibra sem alterar a acurácia (não muda a predição top-1). Base metodológica para tornar seus logprobs confiáveis como probabilidades. *Peer-reviewed, ICML 2017; >4.000 citações.*

---

## (c) Evidências CONTRA / que complicam

**RLHF/alinhamento degrada a calibração de logprobs — GPT-4 Technical Report (2023) e Tian et al. (2023).** O próprio relatório do GPT-4 afirma explicitamente que **o pós-treino (RLHF) reduz a calibração** que existia no modelo pré-treinado. Tian et al. reforçam: para modelos RLHF (ChatGPT, GPT-4, Claude), as confianças *verbalizadas* costumam ser **melhor calibradas** que as probabilidades de token, reduzindo o ECE relativo em ~50% em TriviaQA/SciQ/TruthfulQA. Ou seja, para modelos alinhados, logprobs podem ser o sinal *pior*. *GPT-4: relatório técnico. Tian et al.: peer-reviewed, EMNLP 2023.*

**MCQ tem vieses estruturais que contaminam a probabilidade — Zheng et al. (2024).** "Large Language Models Are Not Robust Multiple Choice Selectors" (ICLR 2024, Spotlight) mostra *selection bias*: LLMs atribuem massa de probabilidade a priori a certos IDs de opção (preferem "A", etc.), independentemente do conteúdo. A probabilidade da letra reflete parcialmente esse viés de token, não só a crença sobre o conteúdo. Propõem debiasing sem rótulos (PriDe). Implicação direta: sua `exp(logprob)` da letra pode estar medindo, em parte, o viés posicional. *Peer-reviewed, ICLR 2024, arXiv 2309.03882.*

**Ranquear por probabilidade de string é traiçoeiro — Holtzman et al. (2021).** "Surface Form Competition" (EMNLP 2021) mostra que respostas com o mesmo significado competem por massa de probabilidade ("computer" vs "PC"), rebaixando a opção correta. Propõem PMI condicional ao domínio. Relevante quando você compara probabilidade de *sequências* de opção (texto completo) em vez da letra; e é um alerta geral de que "maior probabilidade" ≠ "mais correto". *Peer-reviewed, EMNLP 2021, aclanthology 2021.emnlp-main.564.*

**Probabilidade de sequência é sensível a comprimento — cascatas token-level (Google, 2024).** A softmax de uma sequência gerada penaliza sequências longas (length bias), causando deferimento indiscriminado. Se você usar probabilidade de sequência sem normalizar por comprimento, o sinal de confiança fica poluído por tamanho de resposta. *Preprint, arXiv 2404.10136.*

**Raciocínio/matemática quebra a calibração — literatura de CoT/GSM8K.** Modelos ficam calibrados quando respondem diretamente, mas **descalibram ao raciocinar via chain-of-thought** em matemática (GSM8K, OpenMathInstruct): a resposta de pluralidade costuma estar certa, mas é uma pluralidade fraca — logo a probabilidade do token final é pouco informativa. Isso explica diretamente seu **sinal ≈ nulo em matemática**. *Ex.: "Self-Consistency Boosts Calibration for Math Reasoning", preprint arXiv 2403.09849.*

**Cascatas são limitadas pela qualidade do sinal de roteamento, e confiança de LLM é tipicamente mal calibrada.** A própria literatura de cascatas nota que a qualidade da cascata é *limitada pelo sinal de roteamento* e que scores de confiança de LLM são tipicamente mal calibrados, com modelos grandes facilmente superconfiantes — problema crítico quando a confiança decide se a predição do modelo fraco é confiável. *Contexto em GATEKEEPER (arXiv 2502.19335) e correlatos.*

**Superconfiança persiste e não escala trivialmente — "Mind the Confidence Gap" (2025) e literatura de escala.** Trabalhos recentes documentam superconfiança persistente e efeitos de distratores em MCQ; a relação escala↔calibração é não-monotônica e dependente de tarefa (nem sempre "maior = melhor calibrado"). *Preprint, arXiv 2502.11028.*

---

## (d) Boas práticas recomendadas pela literatura

1. **Meça a probabilidade da letra/primeiro token da opção, não a sequência inteira crua.** Reduz *surface form competition* (Holtzman et al., 2021) e o *length bias* (cascatas Google, 2024). Se comparar textos de opção completos, normalize por comprimento ou use PMI condicional ao domínio.

2. **Trate o viés de seleção de MCQ.** Antes de tratar `exp(logprob)` da letra como confiança, mitigue o viés de ID de opção — p.ex. PriDe (Zheng et al., 2024), permutar/embaralhar a ordem das alternativas, ou marginalizar sobre permutações. Sem isso, parte do seu sinal é viés posicional, não crença.

3. **Recalibre antes de usar limiares.** Aplique *temperature scaling* (Guo et al., 2017) num conjunto de validação; barato, não altera a acurácia, e corrige superconfiança sistemática. Isso ataca diretamente seu ECE ≈ 0,40.

4. **Considere confiança verbalizada e `P(True)`/`P(IK)` como sinais complementares ou alternativos**, sobretudo se migrar para modelos alinhados/RLHF, onde a confiança verbalizada às vezes supera logprobs (Tian et al., 2023; Kadavath et al., 2022). Um ensemble simples de sinais (logprob + verbalizado) tende a ser mais robusto.

5. **Para raciocínio/matemática, prefira consistência amostral (self-consistency / entropia semântica) à probabilidade de token único.** Métodos de consistência costumam superar likelihood em perguntas ambíguas e recuperam calibração em math (self-consistency for math reasoning, 2024; survey de UQ, arXiv 2503.15850). Isso endereça seu ponto cego em matemática.

6. **Separe o que você precisa: discriminação vs. calibração.** Para *roteamento* o que importa é **ordenar** acertos acima de erros (AUC / capacidade de *selective prediction*), não o ECE. Otimize e reporte o limiar por curva risco-cobertura (accuracy-vs-coverage), não por ECE. Recalibração ajuda a escolher o limiar, mas não cria discriminação onde não há.

7. **Use incerteza token-level, não só a média.** Quantis/entropia da distribuição de tokens dão trade-off custo-qualidade melhor que uma média de `exp(logprob)` (cascatas Google, 2024).

---

## (e) Como seus achados se encaixam na literatura

Seus resultados são **coerentes e previstos**, não anômalos:

- **AUC ≈ 0,66 (discriminação fraca-moderada).** Consistente com a existência de sinal real, porém fraco, em MCQ — especialmente sem debiasing de opção e sem recalibração. A literatura de cascatas assume exatamente que o sinal existe mas é ruidoso; por isso investe em refino (token-level, calibração). Para roteamento, 0,66 é utilizável como *tie-breaker*/gate parcial, mas não como decisor isolado forte.

- **ECE ≈ 0,40 (forte superconfiança).** Bate com (i) Guo et al. (2017), redes modernas são superconfiantes; (ii) evidência de superconfiança persistente em modelos pequenos e o padrão não-monotônico de escala; (iii) o fato de Gemma-3-1B ser instruction-tuned (alinhamento tende a degradar calibração de logprob, como no GPT-4 Report e Tian et al.). Crucial: **ECE alto não impede roteamento** — ele impede interpretar `exp(logprob)` como probabilidade literal. Temperature scaling deve reduzir muito esse ECE.

- **Sinal ≈ nulo em matemática.** Diretamente previsto pela literatura de CoT/GSM8K: raciocínio descalibra e a probabilidade do token final vira pluralidade fraca. Recomendação: em itens de matemática, troque o sinal por self-consistency (votação sobre múltiplas amostragens) ou escale de tier por *default* (tratar matemática como "sempre difícil" para o modelo pequeno).

- **Modelo pequeno (1B).** A confiança de modelos grandes bem-calibrados (Kadavath, GPT-4 pré-treino) **não se transfere** automaticamente para 1B. Seu regime é o de "sinal fraco + superconfiante", que é justamente onde as mitigações (debiasing de opção, temperature scaling, self-consistency, sinais complementares) têm maior retorno.

**Síntese para o projeto:** manter logprobs como sinal de gate é defensável, mas trate-o como *entrada bruta a ser refinada*, não como confiança pronta. Ganhos esperados de maior impacto, na ordem: (1) recalibrar (temperature scaling) e escolher limiar por curva risco-cobertura; (2) debiasing de opção; (3) política especial para matemática (self-consistency ou escalonamento default); (4) avaliar confiança verbalizada/`P(True)` como sinal complementar.

---

## (f) Referências completas

**Peer-reviewed**

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML 2017. arXiv: https://arxiv.org/abs/1706.04599
- Holtzman, A., West, P., Shwartz, V., Choi, Y., & Zettlemoyer, L. (2021). *Surface Form Competition: Why the Highest Probability Answer Isn't Always Right.* EMNLP 2021. ACL Anthology: https://aclanthology.org/2021.emnlp-main.564/
- Tian, K., Mitchell, E., Zhou, A., Sharma, A., Rafailov, R., Yao, H., Finn, C., & Manning, C. D. (2023). *Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback.* EMNLP 2023. ACL Anthology: https://aclanthology.org/2023.emnlp-main.330/
- Zheng, C., Zhou, H., Meng, F., Zhou, J., & Huang, M. (2024). *Large Language Models Are Not Robust Multiple Choice Selectors.* ICLR 2024 (Spotlight). OpenReview: https://openreview.net/forum?id=shr9PXz7T0 · arXiv: https://arxiv.org/abs/2309.03882
- Chen, L., Zaharia, M., & Zou, J. (2024). *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance.* TMLR 2024. arXiv: https://arxiv.org/abs/2305.05176

**Preprints / relatórios técnicos**

- Kadavath, S., Conerly, T., Askell, A., et al. (2022). *Language Models (Mostly) Know What They Know.* Anthropic. arXiv: https://arxiv.org/abs/2207.05221
- OpenAI (2023). *GPT-4 Technical Report.* arXiv: https://arxiv.org/abs/2303.08774
- Gupta, N., et al. / Google Research (2024). *Language Model Cascades: Token-level uncertainty and beyond.* arXiv: https://arxiv.org/abs/2404.10136
- (Autoria coletiva, 2024). *Self-Consistency Boosts Calibration for Math Reasoning.* arXiv: https://arxiv.org/abs/2403.09849
- *Mind the Confidence Gap: Overconfidence, Calibration, and Distractor Effects in Large Language Models* (2025). arXiv: https://arxiv.org/abs/2502.11028
- *GATEKEEPER: Improving Model Cascades Through Confidence Tuning* (2025). arXiv: https://arxiv.org/abs/2502.19335
- *Uncertainty Quantification and Confidence Calibration in Large Language Models: A Survey* (2025). arXiv: https://arxiv.org/abs/2503.15850

**Métodos correlatos citados (auto-consistência / entropia semântica)** — referências fundacionais frequentemente usadas nesta linha: Wang et al., *Self-Consistency Improves Chain of Thought Reasoning* (ICLR 2023); Kuhn et al., *Semantic Uncertainty* (ICLR 2023); Farquhar et al., *Detecting hallucinations using semantic entropy* (Nature, 2024). Recomenda-se confirmar os links antes de citar formalmente.

---

## Verificação de credibilidade das fontes

| Fonte | Tipo | Venue | Credibilidade |
|---|---|---|---|
| Guo et al. 2017 | Peer-reviewed | ICML 2017 | Muito alta — >4.000 citações; padrão de calibração |
| Holtzman et al. 2021 | Peer-reviewed | EMNLP 2021 | Alta — venue top de NLP; muito citado |
| Tian et al. 2023 | Peer-reviewed | EMNLP 2023 | Alta — venue top; autores de Stanford (Manning, Finn) |
| Zheng et al. 2024 | Peer-reviewed | ICLR 2024 **Spotlight** | Alta — Spotlight indica destaque do comitê |
| FrugalGPT (Chen et al.) | Peer-reviewed | TMLR 2024 (após preprint 2023) | Alta — journal com revisão; muito citado |
| Kadavath et al. 2022 | Preprint (relatório Anthropic) | arXiv | Alta influência (~400+ citações), mas **não peer-reviewed** |
| GPT-4 Technical Report | Relatório técnico | arXiv/OpenAI | Extremamente citado, mas **não peer-reviewed**; detalhes limitados por opção da OpenAI |
| Cascades token-level (Google) | Preprint | arXiv | Google Research; sólido, mas confirmar status de publicação |
| Self-Consistency for Math / Mind the Confidence Gap / GATEKEEPER / Survey UQ | Preprints | arXiv | Úteis como evidência corroborante; **não peer-reviewed** — usar com cautela |

**Observações de verificação:** as três fontes que sustentam o núcleo do veredito favorável em MCQ (Kadavath; GPT-4 Report) são *não* peer-reviewed, embora de alta influência e alinhadas entre si — o que reforça a confiança apesar do status de preprint. As fontes que *complicam* a tese (Zheng, Holtzman, Tian, Guo) são todas peer-reviewed em venues top (ICLR/EMNLP/ICML), o que dá peso extra às ressalvas. Recomenda-se, para uso formal em uma dissertação/artigo, checar contagens de citação atualizadas (Google Scholar / Semantic Scholar) e confirmar o venue final dos preprints antes de citar.
