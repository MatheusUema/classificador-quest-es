# Classificador de Dificuldade e Roteamento de LLMs para Educacao

## Documento Conceitual — Pesquisa, Pipeline e Elasticidade

---

## 1. Contexto e Motivacao

Este projeto implementa um sistema de **roteamento inteligente de LLMs** para um assistente educacional voltado a escolas brasileiras (ensino fundamental e medio). O problema central: modelos de linguagem locais (on-device) sao leves e privados mas limitados; modelos cloud sao capazes mas dependem de conectividade e custam dinheiro. A pergunta e: **como decidir automaticamente qual modelo deve responder cada pergunta?**

### Artigos-base

1. **RouteLLM** (2024) — Propoe um classificador binario treinado com dados de preferencia que roteia entre um modelo fraco e um forte. Usa embeddings + MLP. Nosso projeto estende para 3 tiers (local/servidor/cloud) com score continuo [0,1].

2. **AIED Unplugged — Paper 901** (Elasticidade Tecnica) — Define 5 principios para IA educacional em contextos desconectados:
   - **Conformity**: funcionar dentro das restricoes reais da escola
   - **Disconnect**: operar sem internet
   - **Proxy**: professor como mediador quando IA nao alcanca
   - **Multi-user**: varios alunos compartilham dispositivo
   - **Unskillfulness**: funcionar sem conhecimento tecnico

3. **AIED Unplugged — Paper 1750** (Elasticidade Pedagogica) — Define 3 modos de resposta pedagogica:
   - **Direct-answer**: LLM responde diretamente
   - **Scaffolded-support**: LLM guia com perguntas socraticas
   - **Mediated/Deferred**: LLM nao consegue responder; encaminha ao professor

   A **Tabela 4** do paper mapeia condicoes locais (conectividade, capacidade do modelo, presenca do professor) para esses modos.

---

## 2. Arquitetura Hibrida de Roteamento

### Insight fundamental

O classificador de dificuldade baseado em texto (features superficiais como n_palavras, conectivos) tem **R² = 0.15** — ou seja, features de texto sao proxies fracos da dificuldade real de uma questao. Um texto curto pode ser extremamente dificil ("Resolva: integral de e^(x^2) dx") e um texto longo pode ser trivial.

Isso levou a uma mudanca arquitetural: o classificador ONNX funciona apenas como **pre-filtro para casos extremos**, enquanto a **confianca do proprio modelo local** e o sinal principal de roteamento.

### Decisao: classificador model-agnostic + confianca model-specific

> "O classificador precisa ser compativel com a LLM local escolhida. Se mudarmos a LLM local no futuro, suas respostas podem ser mais confiantes."

Separamos em duas camadas:

| Camada | O que mede | Depende do modelo? |
|---|---|---|
| **ONNX pre-filter** | Complexidade do texto da questao | Nao — e model-agnostic |
| **Confianca da LLM** | Capacidade do modelo de responder aquela questao | Sim — calibrado por modelo |

O pre-filter ONNX classifica a **complexidade da questao** (nao a dificuldade para o modelo). A confianca da LLM mede se **aquele modelo especifico** consegue responder bem. Trocar o modelo local so exige recalibrar os thresholds de confianca, nao retreinar o classificador.

---

## 3. Pipeline de Dados — Classificador ONNX

### Fonte de dados

**ENEM (2020-2024)** — Microdados do INEP:
- `ITENS_PROVA_<ANO>.csv`: parametros IRT calibrados pelo INEP
- `PROVAS E GABARITOS/*.pdf`: texto das questoes

O parametro **B (dificuldade IRT)** e calibrado sobre milhoes de respondentes e mede dificuldade para humanos. Convertemos para score via sigmoide: `difficulty_score = 1 / (1 + exp(-B))`.

### Pipeline (pipeline.py)

```
PDF (P1 AMARELO) ──> extrair_questoes_pdf() ──> texto por CO_POSICAO
                                                        |
CSV (ITENS_PROVA) ──> carregar_itens_ano() ──────> join por CO_POSICAO
                                                        |
                                                  extrair_features()
                                                        |
                                                  dataset_enem_dificuldade.csv
```

**Join**: CO_POSICAO (numero da questao no caderno) liga o texto do PDF ao parametro B do CSV. Filtramos por TX_COR = "AMARELA" e menor CO_PROVA por area (= P1 aplicacao regular).

### Features (11 dimensoes)

| Feature | Tipo | Descricao |
|---|---|---|
| n_palavras | int | Total de palavras |
| media_palavras_por_sentenca | float | Comprimento medio das sentencas |
| pct_palavras_longas | float | % de palavras com >8 caracteres |
| n_conectivos_logicos | int | Contagem de conectivos (portanto, logo, etc.) |
| tem_negacao | 0/1 | Presenca de negacao (nao, exceto, etc.) |
| tem_formula_ou_numero | 0/1 | Presenca de formulas, numeros, simbolos |
| tem_imagem | 0/1 | Proxy: texto < 30 palavras = provavel figura |
| area_linguagens | 0/1 | One-hot area LC |
| area_humanas | 0/1 | One-hot area CH |
| area_natureza | 0/1 | One-hot area CN |
| area_matematica | 0/1 | One-hot area MT |

### Modelo treinado (train.py)

- **GradientBoostingRegressor** (200 estimators, max_depth=4, lr=0.05)
- Split 80/20 estratificado por area
- MAE = 0.1069, R² = 0.1492
- Exportado para `.joblib` (Python/servidor) e `.onnx` (mobile)
- Validacao ONNX: diferenca maxima vs joblib = 7.34e-08

### Papel no sistema final

O classificador ONNX **nao** decide o roteamento sozinho. Ele atua como pre-filtro rapido:
- Score < θ_easy → certamente local (sem precisar rodar a LLM para avaliar confianca)
- Score > θ_hard → certamente cloud (questao muito complexa para qualquer LLM local)
- Score no meio → **zona cinza** — roda a LLM local e avalia confianca da resposta

Os thresholds θ_easy e θ_hard sao configuraveis por escola sem retreino.

### SAEB como fonte adicional

SAEB (ensino fundamental e medio) pode complementar o dataset. Implicacoes:
- SAEB cobre materias do EF/EM que o ENEM nao contempla diretamente
- O SAEB nao tem a mesma distribuicao de dificuldade (faixa B mais restrita)
- Seria necessario normalizar as escalas IRT entre ENEM e SAEB
- Beneficio principal: aumentar cobertura de conteudos de matematica e portugues basicos

### Questoes com imagem

Questoes com figura dominante geram texto curto no PDF (< 30 palavras). O pipeline marca `tem_imagem=1` como proxy. No roteamento, essas questoes tendem a ir para cloud naturalmente porque:
1. O pre-filter ONNX recebe features pobres → score incerto → zona cinza
2. A LLM local (texto-only) nao "ve" a imagem → confianca baixa → escalona

---

## 4. Sinal de Confianca — Estrategia de Validacao

### O problema

Nenhuma das bibliotecas de inferencia local no Android expoe logprobs:
- **MediaPipe LLM Inference**: `generateResponse()` retorna `String` — deprecated, substituido por LiteRT-LM
- **llama.rn**: wrapper JS nao expoe logprobs (llama.cpp por baixo suporta)
- **llama.cpp via JNI**: acesso direto a `llama_get_logits()` — possivel mas esforco alto (NDK/CMake/C++)

### Decisao arquitetural: llama.cpp SERVER na rede local

Em vez de migrar o runtime on-device para llama.cpp JNI (estimativa: 2-3 semanas de C++), utilizamos o **servidor HTTP embutido do llama.cpp** (`llama-server`) rodando em uma maquina na rede local da escola. Isso porque:

1. **Objetivo imediato e validar hipoteses**, nao construir produto final
2. O endpoint `/completion` do llama-server **ja retorna logprobs nativamente** — zero codigo custom
3. Esforco de integracao no app: apenas um HTTP client (Retrofit/OkHttp) — ~2-3 dias
4. Alinha perfeitamente com o tier "servidor" da arquitetura de 3 tiers
5. MediaPipe local continua funcionando para o tier offline (sem tocar no que ja funciona)

**A migracao JNI nao e descartada** — ela se torna uma evolucao futura informada por dados. Se os logs de pesquisa comprovarem que logprobs melhoram significativamente o roteamento, ai justifica-se o investimento para trazer logprobs para on-device.

### Arquitetura de validacao

```
Tier LOCAL (offline)     → MediaPipe (sem confianca, usa heuristicas de output)
Tier SERVIDOR (rede)     → llama.cpp server (COM logprobs via API HTTP)
Tier CLOUD (internet)    → Firebase AI Logic / Gemini
```

O tier servidor e onde validamos a hipotese de confianca. O tier local e o baseline offline-first. A comparacao entre roteamento com confianca (servidor) vs heuristica (local) gera os dados da pesquisa.

### Metrica de confianca

O endpoint `/completion` do llama-server retorna `completion_probabilities` com a probabilidade de cada token gerado:

```json
{
  "content": "A fotossintese e o processo...",
  "completion_probabilities": [
    {"content": "A", "probs": [{"tok_str": "A", "prob": 0.92}]},
    {"content": " foto", "probs": [{"tok_str": " foto", "prob": 0.87}]}
  ]
}
```

A metrica de confianca:

```
confianca = media(prob do token escolhido para cada posicao)
```

**Thresholds de confianca** (calibrados por modelo via benchmark embutido):
- confianca > θ_alta (0.7) → resposta confiavel → entrega diretamente (direct-answer)
- confianca < θ_baixa (0.3) → modelo inseguro → escalona para cloud
- confianca entre θ_baixa e θ_alta → entrega com modo scaffolded (pedagogico)

### Heuristicas para tier local (sem logprobs)

Quando o servidor nao esta disponivel e o app opera apenas com MediaPipe local:
- Resposta muito curta (< 20 chars) → baixa confianca
- Repeticao de frases (loop detectado) → baixa confianca
- Timeout proximo do limite → baixa confianca
- Resposta generica sem relacao com a pergunta → baixa confianca

Essas heuristicas sao o "modo degradado" — funcionam como circuit breaker para falhas obvias mas nao oferecem gradiente fino de confianca. A comparacao entre heuristicas (local) e logprobs (servidor) e um dos outputs da pesquisa.

### Roadmap de confianca

| Fase | Abordagem | Onde roda | Esforco | Objetivo |
|---|---|---|---|---|
| **v1 (agora)** | Heuristicas de output | On-device (MediaPipe) | Baixo | Baseline offline |
| **v2 (validacao)** | Logprobs via HTTP | Servidor llama.cpp | Baixo | Validar hipotese |
| **v3 (futuro)** | Logprobs via JNI | On-device (llama.cpp) | Alto | Producao on-device |

A transicao v2→v3 so acontece se os dados de pesquisa justificarem.

---

## 5. Fluxo Completo de Roteamento (Ponta a Ponta)

```
                                    +------------------+
                                    |   Input do aluno |
                                    +--------+---------+
                                             |
                                    +--------v---------+
                                    | Pre-filter ONNX  |
                                    | (complexidade do  |
                                    |  texto)           |
                                    +---+----+----+----+
                                        |    |    |
                               < θ_easy |    |    | > θ_hard
                                        |    |    |
                              +---------+    |    +----------+
                              |              |               |
                    +---------v--+    +------v-------+    +--v---------+
                    | LLM LOCAL  |    | LLM LOCAL    |    | CLOUD LLM  |
                    | (resposta  |    | (gera + avalia|    | (resposta  |
                    |  direta)   |    |  confianca)  |    |  completa) |
                    +-----+------+    +---+---+------+    +-----+------+
                          |               |   |                 |
                          |        alta   |   | baixa           |
                          |               |   |                 |
                    +-----v------+  +-----v-+ +---v--------+   |
                    | direct-    |  | direct-| | scaffolded |   |
                    | answer     |  | answer | | ou cloud   |   |
                    +------------+  +--------+ +------+-----+   |
                                                      |         |
                                               +------v---------v---+
                                               | Modo Pedagogico    |
                                               | (direct/scaffolded/|
                                               |  mediated)         |
                                               +--------+-----------+
                                                        |
                                               +--------v-----------+
                                               | Log de pesquisa    |
                                               | (questao, score,   |
                                               |  rota, confianca,  |
                                               |  modo, latencia)   |
                                               +--------------------+
```

---

## 6. Modos Pedagogicos e Elasticidade

### Tres modos de resposta (do Paper 1750)

| Modo | Quando usar | Comportamento |
|---|---|---|
| **Direct-answer** | Confianca alta OU cloud | Responde diretamente com explicacao |
| **Scaffolded-support** | Confianca media da LLM local | Guia com perguntas socraticas, nao entrega resposta |
| **Mediated/Deferred** | Sem conectividade + confianca baixa | Registra duvida para o professor revisar depois |

### Mapeamento com TutorMode do app

O app ja implementa 4 TutorModes (EXPLAIN, HINT, SUMMARY, REVIEW). O mapeamento com os 3 modos pedagogicos:

| Modo pedagogico | TutorMode correspondente | Trigger |
|---|---|---|
| Direct-answer | EXPLAIN, SUMMARY, REVIEW | Confianca alta ou cloud |
| Scaffolded-support | HINT | Confianca media (local) |
| Mediated/Deferred | (novo: DEFERRED) | Offline + confianca baixa |

---

## 7. Alinhamento com Elasticidade

### Elasticidade Tecnica (Paper 901)

O sistema **expande** sem degradar o baseline:

| Cenario | Tier disponivel | Comportamento |
|---|---|---|
| **Unplugged** (sem internet, sem servidor) | Apenas local | LLM local + modos pedagogicos adaptativos |
| **Plugged parcial** (rede local) | Local + servidor | Questoes dificeis vao para servidor LLM |
| **Plugged total** (internet) | Local + servidor + cloud | Questoes complexas vao para cloud |

A **configuracao e por YAML** sem recompilar o app:
```yaml
# Escola rural sem internet
tiers:
  local: { enabled: true, model: "gemma-3-1b-Q4_K_M.gguf" }
  server: { enabled: false }
  cloud: { enabled: false }
fallback_mode: "mediated"

# Escola urbana com lab
tiers:
  local: { enabled: true, model: "gemma-3-1b-Q4_K_M.gguf" }
  server: { enabled: true, url: "http://192.168.1.100:8080" }
  cloud: { enabled: true, provider: "firebase" }
```

### Elasticidade Pedagogica (Paper 1750)

O modo de resposta se adapta a capacidade real do sistema:

| Condicao local | Modo automatico | Justificativa |
|---|---|---|
| Local + confianca alta | Direct-answer | Modelo sabe responder |
| Local + confianca media | Scaffolded-support | Modelo inseguro → guia em vez de arriscar |
| Offline + confianca baixa | Mediated/Deferred | Registra para professor |
| Cloud disponivel | Direct-answer (cloud) | Modelo capaz responde |
| Privacidade ativa + local | Direct-answer (local) | Dados nunca saem do device |

### Alinhamento com os 5 principios AIED Unplugged

| Principio | Como o sistema atende |
|---|---|
| **Conformity** | Config YAML por escola; thresholds ajustaveis; modelos GGUF de varios tamanhos |
| **Disconnect** | Tier local funciona 100% offline; modo mediated registra duvidas para depois |
| **Proxy** | Modo mediated/deferred encaminha ao professor; scaffolded reduz dependencia da IA |
| **Multi-user** | Sessoes independentes; logs por aluno; modelo local compartilhado sem conflito |
| **Unskillfulness** | Roteamento automatico; professor so configura YAML ou toggles na UI |

---

## 8. Log de Pesquisa

Cada interacao registra:

```json
{
  "timestamp": "2026-06-24T14:30:00Z",
  "session_id": "abc-123",
  "question_text": "O que e fotossintese?",
  "onnx_score": 0.42,
  "route_decision": "local_with_confidence",
  "confidence_score": 0.87,
  "confidence_method": "logprobs_mean",
  "final_tier": "local",
  "pedagogical_mode": "direct-answer",
  "tutor_mode": "EXPLAIN",
  "latency_ms": 1230,
  "model_id": "gemma-3-1b-Q4_K_M",
  "connectivity": "offline",
  "device_info": "Pixel 8 Pro"
}
```

Esses logs alimentam a pesquisa: permitem analisar se o roteamento esta correto, medir a qualidade das respostas por tier, e calibrar thresholds com dados reais.

---

## 9. Alternativas Consideradas e Contra-Argumentos

Ao longo do projeto, diversas alternativas foram avaliadas antes de convergir na arquitetura hibrida (ONNX pre-filter + llama.cpp com logprobs). Esta secao documenta cada alternativa, por que foi considerada, e por que foi descartada ou relegada a papel secundario.

### 9.1 Classificador ONNX como roteador unico (sem confianca da LLM)

**O que seria**: O classificador treinado com features de texto decide sozinho para qual tier enviar a questao, usando thresholds fixos sobre o score [0,1].

**Por que foi considerado**: Abordagem mais simples, sem dependencia de logprobs, funciona com qualquer backend de inferencia.

**Por que foi descartado**:
- R² = 0.1492 — features superficiais de texto (n_palavras, conectivos, etc.) sao proxies muito fracos da dificuldade real. O score se aglomera na faixa [0.58-0.89] para questoes muito diferentes.
- Nos testes com 10 exemplos, apenas 3/10 foram roteados corretamente.
- O classificador mede **complexidade textual**, nao **dificuldade para o modelo**. Uma questao curta como "Resolva: integral de e^(x^2) dx" tem poucas palavras mas e extremamente dificil.
- O RouteLLM original usa embeddings de modelos de linguagem como features, nao features de texto superficiais — portanto, nosso classificador com features leves nao replica fielmente a abordagem do paper.

**Papel residual**: Pre-filtro para casos extremos (texto trivial ou claramente complexo), evitando rodar a LLM local desnecessariamente.

### 9.2 Consistency check (duas geracoes)

**O que seria**: Para cada pergunta, gerar a resposta 2x com seeds/temperaturas diferentes. Se as respostas divergem, o modelo esta inseguro → escalonar.

**Por que foi considerado**: Funciona com qualquer biblioteca que retorne texto (MediaPipe, llama.rn, etc.), sem precisar de logprobs.

**Por que foi descartado como solucao principal**:
- **Latencia 2x**: Em dispositivos moveis com modelos de 1-3B, cada geracao leva 5-15 segundos. Duplicar isso e inaceitavel para UX educacional — o aluno nao pode esperar 30s por uma resposta.
- **Custo de bateria**: Inferencia local e o componente que mais consome energia. Dobrar o uso e incompativel com dispositivos escolares de baixo custo que frequentemente tem baterias pequenas.
- **Falso senso de seguranca**: Modelos pequenos podem ser "consistentemente errados" — gerar a mesma resposta incorreta nas duas tentativas com alta confianca aparente. A concordancia entre geracoes nao garante corretude.
- **Threshold arbitrario**: Como definir "divergencia"? Comparacao textual exata e muito estrita (reformulacoes sao naturais); comparacao semantica exigiria outro modelo.

**Papel residual**: Possivel como metrica complementar em cenarios onde logprobs nao estao disponiveis (fallback v1).

### 9.3 Temperatura como proxy de confianca

**O que seria**: Gerar com temperatura muito baixa (0.1) e comparar com geracao normal (0.7). Se as respostas divergem, o modelo esta na "fronteira de decisao" entre tokens → incerteza.

**Por que foi considerado**: Variacao da consistency check, potencialmente mais informativa porque temperatura baixa forca o modelo a seguir o caminho de maior probabilidade.

**Por que foi descartado**:
- Mesmos problemas de latencia e bateria da consistency check (duas geracoes).
- Temperatura 0.1 nao elimina variabilidade — apenas a reduz. Ainda sofre do problema de "consistentemente errado".
- A interpretacao da divergencia e ambigua: respostas diferentes com temperaturas diferentes podem simplesmente refletir que o modelo tem multiplas respostas validas, nao que esta inseguro.
- Logprobs dao acesso direto a distribuicao de probabilidade em **cada token**, sem precisar de geracao duplicada — e estritamente mais informativo.

### 9.4 Padroes de output (heuristicas de texto da resposta)

**O que seria**: Detectar sinais de falha na resposta gerada sem logprobs:
- Resposta muito curta (< 20 chars)
- Repeticao de frases (loop)
- Timeout proximo do limite
- Resposta generica / off-topic

**Por que foi considerado**: Custo zero (nenhuma geracao extra), implementavel com qualquer backend, sem dependencia de API especifica.

**Por que foi descartado como solucao principal**:
- **Cobertura limitada**: So detecta falhas **catastroficas** (modelo travou, gerou lixo). Nao detecta respostas fluentes mas factualmente erradas — que e o caso mais perigoso em contexto educacional.
- **Falsos positivos**: Respostas curtas podem ser corretas ("A capital do Brasil e Brasilia."). O threshold de comprimento e arbitrario e varia por tipo de pergunta.
- **Sem gradiente**: Retorna binario (ok/falha), nao um score continuo de confianca. Nao permite a zona intermediaria que ativa o modo scaffolded.
- **Fragil**: Cada heuristica requer tuning manual e quebra com modelos diferentes. Mudar o modelo local invalida os thresholds.

**Papel residual**: Camada v1 imediata enquanto a migracao para llama.cpp nao esta pronta. Util como "circuit breaker" para falhas obvias mesmo apos ter logprobs.

### 9.5 Permanecer com MediaPipe LLM Inference

**O que seria**: Manter a implementacao atual com MediaPipe, sem migrar.

**Por que foi considerado**: Zero esforco de migracao, codigo ja funcional e testado.

**Por que foi descartado**:
- **API deprecated**: Google colocou MediaPipe LLM Inference em modo manutencao. O substituto oficial e LiteRT-LM. Permanecer numa API deprecated significa acumular divida tecnica sem receber bugfixes ou novas features.
- **Sem logprobs**: `session.generateResponse()` retorna apenas `String`. Nao ha plano publico de adicionar logprobs. Sem logprobs, o sinal de confianca fica limitado a heuristicas frageis (secao 9.4).
- **Ecossistema fechado**: Modelos precisam ser convertidos para formato proprietario `.task`. Poucos modelos sao oficialmente suportados (Gemma 2B, Gemma-2 2B, Phi-2). Trocar de modelo e trabalhoso — fere a elasticidade tecnica.
- **Sem controle de sampling**: Nao e possivel ajustar parametros avancados de geracao (repetition penalty, min_p, etc.) que podem ser relevantes para qualidade pedagogica.

### 9.6 Migrar para LiteRT-LM (substituto oficial do MediaPipe)

**O que seria**: Migrar para a API que o Google indica como substituta do MediaPipe LLM Inference.

**Por que foi considerado**: Caminho "oficial" de migracao, potencialmente melhor integracao com ecossistema Android, possivel suporte a GPU via delegates.

**Por que foi descartado (por enquanto)**:
- **API muito recente**: LiteRT-LM estava em estagio inicial em meados de 2025. Documentacao escassa, poucos exemplos, comunidade pequena.
- **Sem evidencia de logprobs**: Ate onde investigado, LiteRT-LM tambem expoe uma API de alto nivel sem acesso a logits internos. Sem logprobs, nao resolve o problema central de confianca.
- **Lock-in Google**: Assim como MediaPipe, depende de formatos e ferramentas do ecossistema Google. Se o Google descontinuar novamente (como fez com MediaPipe), outra migracao sera necessaria.
- **llama.cpp e mais maduro para nosso caso**: Comunidade enorme, formato GGUF universal, API C estavel com acesso completo aos logits, e centenas de modelos compativeis no HuggingFace.

**Quando reconsiderar**: Se LiteRT-LM adicionar API de logprobs e atingir maturidade comparavel ao llama.cpp, pode ser vantajoso pela melhor integracao com GPU Android.

### 9.7 App do zero com llama.cpp (em vez de migrar o existente)

**O que seria**: Criar um novo projeto Android do zero usando llama.cpp, em vez de modificar o voiceassistant existente.

**Por que foi considerado**: Evita lidar com codigo legado, permite arquitetura "ideal" desde o inicio.

**Por que foi descartado**:
- **Retrabalho massivo**: O app atual ja tem UI funcional (Jetpack Compose), reconhecimento de voz (STT), sintese de fala (TTS), sistema de chat, modos pedagogicos (TutorMode), InferenceRouter com logica testada, integracao com Firebase Cloud, e gerenciamento de modelo local.
- **Arquitetura ja isolada**: A `LocalInferenceService` e uma interface limpa com 5 metodos. O `ServiceModule` Hilt permite trocar a implementacao com uma unica linha de binding. A migracao afeta apenas a camada de inferencia local — todo o resto (UI, voz, router, cloud) permanece intacto.
- **Escopo da pesquisa**: Reescrever o app inteiro expandiria o escopo alem do necessario para a pesquisa. O objetivo e validar o roteamento com confianca, nao construir um app perfeito.
- **Tempo**: Estimativa de meses vs 2-3 semanas para a migracao.

### 9.8 Usar embedding-based classifier (RouteLLM fiel)

**O que seria**: Replicar a abordagem exata do RouteLLM — usar embeddings de um modelo de linguagem como features para o classificador, em vez de features de texto superficiais.

**Por que foi considerado**: Resolveria o problema do R² baixo, pois embeddings capturam semantica, nao apenas superficie textual.

**Por que foi descartado (por enquanto)**:
- **Custo computacional no device**: Gerar embeddings exige rodar um modelo (mesmo que pequeno, como MiniLM ~30MB) antes de cada classificacao. No fluxo atual, o pre-filter ONNX roda em <5ms com 11 features numericas. Adicionar embedding aumentaria para 50-200ms + consumo de memoria.
- **Complexidade de pipeline**: Exigiria um segundo modelo (embedding) alem do LLM principal, complicando o gerenciamento de modelos no device.
- **O pre-filter nao precisa ser perfeito**: Na arquitetura hibrida, o ONNX so filtra extremos. A confianca da LLM (logprobs) e o sinal que importa. Investir em melhorar o pre-filter tem retorno decrescente enquanto os logprobs resolvem o caso geral.
- **Dados de treino**: O RouteLLM usa dados de preferencia humana (chatbot arena). Nos nao temos esses dados — temos IRT do ENEM, que mede dificuldade humana (nao dificuldade para LLM). Embeddings com labels errados nao resolvem o problema.

**Quando reconsiderar**: Apos coletar dados reais de "a LLM local acertou/errou" via logs de pesquisa, retreinar um classificador com embeddings + esses labels seria uma evolucao natural (secao 10.2).

### 9.9 Servidor LLM intermediario (rede local) como tier unico

**O que seria**: Em vez de 3 tiers, usar apenas 2: dispositivo local (para UI/cache) + servidor na rede local da escola (para toda inferencia LLM).

**Por que foi considerado**: Simplifica o roteamento; servidor na rede local pode rodar modelos maiores (7B-13B) com GPU; latencia baixa via LAN.

**Por que foi descartado como arquitetura unica**:
- **Viola principio Disconnect**: Se o servidor cair ou a rede local falhar, o sistema para completamente. Em escolas publicas brasileiras, quedas de rede/energia sao frequentes.
- **Ponto unico de falha**: Diferente do cloud (alta disponibilidade), um servidor local e uma maquina fisica que pode superaquecer, travar, ou ser desligado por engano.
- **Custo de infraestrutura**: Nem toda escola tem hardware para rodar um servidor LLM. O principio Conformity exige funcionar com o que a escola tem.
- **Nao escala para muitos alunos**: Um servidor com GPU atende ~5-10 requisicoes simultaneas. Em uma sala com 30 alunos, ha contenção.

**Papel no sistema**: Tier intermediario opcional. A arquitetura de 3 tiers permite usar o servidor quando disponivel sem depender dele.

### Resumo da analise de alternativas

| Alternativa | Vantagem principal | Razao de descarte | Status |
|---|---|---|---|
| ONNX como roteador unico | Simples, sem logprobs | R²=0.15, 3/10 acertos | Pre-filtro apenas |
| Consistency check | Funciona com qualquer backend | Latencia 2x, consistentemente errado | Fallback complementar |
| Temperatura como proxy | Mais informativo que consistency | Mesma latencia 2x, interpretacao ambigua | Descartado |
| Padroes de output | Custo zero | So detecta falhas catastroficas | v1 temporaria |
| Permanecer MediaPipe | Zero esforco | Deprecated, sem logprobs, lock-in | Descartado |
| Migrar para LiteRT-LM | Caminho oficial Google | Imaturo, provavelmente sem logprobs | Monitorar |
| App do zero | Sem legado | Retrabalho massivo, desnecessario | Descartado |
| Embedding classifier | R² melhor | Custo no device, labels errados | Evolucao futura |
| Servidor unico | Modelos maiores | Viola Disconnect, ponto de falha | Tier opcional |

---

## 10. Evolucoes Futuras

1. **Embedding-based classifier** — Trocar features de texto por embeddings de um modelo leve (e.g., sentence-transformers) para melhorar o pre-filter ONNX
2. **LLM-performance labels** — Retreinar o classificador usando "a LLM local conseguiu responder corretamente?" como label, em vez de IRT (que mede dificuldade humana)
3. **Calibracao online** — Usar os logs de pesquisa para ajustar thresholds automaticamente
4. **SAEB dataset** — Expandir cobertura para conteudos de EF/EM
5. **Benchmark embutido** — 50 questoes calibradas rodadas na primeira execucao para definir thresholds por modelo
