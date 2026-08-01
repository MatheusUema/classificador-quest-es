# Plano de Implementacao — Tier Servidor (llama.cpp server) + Roteamento com Confianca

## Briefing para sessao de desenvolvimento

Este documento descreve as mudancas necessarias no app `voiceassistant` para adicionar um tier de servidor local (llama.cpp HTTP server) com logprobs, evoluir o InferenceRouter para roteamento baseado em confianca, e adicionar log de pesquisa. **Nao ha migracao do runtime local** — MediaPipe permanece como tier offline.

**Repositorio**: `C:\Users\mathe\AndroidStudioProjects\voiceassistant`
**Linguagem**: Kotlin
**Build**: Gradle (sem NDK/CMake)
**DI**: Hilt
**Estimativa**: 3-4 dias

---

## Arquitetura Atual do App

### Fluxo de inferencia

```
ChatViewModel
  -> SendMessageUseCase
    -> InferenceRepository (interface)
      -> InferenceRouter (implementacao, @Singleton)
        -> LocalInferenceService (interface)
          -> MediaPipeLocalInferenceService (implementacao)
        -> CloudInferenceService (interface)
          -> FirebaseCloudInferenceService (implementacao)
```

### Interfaces existentes (NAO modificar)

```kotlin
// LocalInferenceService.kt
interface LocalInferenceService {
    suspend fun generate(prompt: String): String
    val isModelLoaded: Boolean
    val isAvailable: Boolean
    suspend fun loadModel(modelPath: String)
    suspend fun warmup(prompt: String = "Ola")
    fun unloadModel()
}

// CloudInferenceService.kt (inferida — mesmo padrao)
interface CloudInferenceService {
    suspend fun generate(prompt: String): String
    val isAvailable: Boolean
}
```

### Modelos de dados existentes

```kotlin
// InferenceResult.kt
data class InferenceResult(
    val text: String,
    val source: InferenceSource,
    val latencyMs: Long
)

// InferenceSource.kt
enum class InferenceSource { LOCAL, CLOUD, FALLBACK }

// InferenceRequest.kt
data class InferenceRequest(
    val prompt: String,
    val sessionId: String,
    val conversationHistory: List<ChatMessage> = emptyList(),
    val complexity: PromptComplexity = PromptComplexity.SIMPLE,
    val tutorMode: TutorMode = TutorMode.EXPLAIN
)

enum class PromptComplexity { SIMPLE, MODERATE, COMPLEX }
```

### Roteamento atual (InferenceRouter.kt)

```kotlin
// Logica pura em resolveRoute():
// 1. privacidade + local → LOCAL
// 2. privacidade sem local → ERROR
// 3. offline + local → LOCAL
// 4. offline sem local → ERROR
// 5. online + COMPLEX + cloud → CLOUD
// 6. online + local → LOCAL_WITH_CLOUD_FALLBACK
// 7. online + cloud (sem local) → CLOUD
// 8. nada → ERROR

enum class RoutingDecision {
    LOCAL, CLOUD, LOCAL_WITH_CLOUD_FALLBACK,
    ERROR_PRIVACY, ERROR_OFFLINE, ERROR_UNAVAILABLE
}
```

### PromptComplexityAnalyzer.kt (pre-filtro existente)

```kotlin
// Classifica por: wordCount, keywords complexos, questionCount
// SIMPLE: < 20 palavras, sem keywords
// MODERATE: 20-50 palavras ou 2 perguntas
// COMPLEX: > 50 palavras ou keywords (explique, compare, calcule, etc.) ou 3+ perguntas
```

### TutorPromptBuilder.kt

Gera prompts adaptados por:
- `TutorMode`: EXPLAIN, HINT, SUMMARY, REVIEW
- `compact`: true (local, prompts curtos) / false (cloud, prompts ricos)
- Historico limitado: 2 mensagens (local) / 10 (cloud)

### ServiceModule.kt (Hilt bindings)

```kotlin
@Binds abstract fun bindLocalInference(impl: MediaPipeLocalInferenceService): LocalInferenceService
@Binds abstract fun bindCloudInference(impl: FirebaseCloudInferenceService): CloudInferenceService
```

---

## O que implementar

### Visao geral das mudancas

```
NOVO:
  ai_server/                          ← novo modulo para tier servidor
    service/ServerInferenceService.kt ← HTTP client para llama-server
    model/ServerConfig.kt             ← URL, timeout, thresholds
    model/ServerResponse.kt           ← parsing da resposta com logprobs

MODIFICAR:
  core/model/InferenceResult.kt       ← adicionar campo confidence
  core/model/InferenceSource.kt       ← adicionar SERVER
  feature_tutor/policy/InferenceRouter.kt ← adicionar tier servidor + logica de confianca
  di/ServiceModule.kt                 ← binding do novo service
  core/model/InferenceRequest.kt      ← (sem mudanca)

NOVO:
  core/logging/RoutingLogger.kt       ← log de pesquisa (Room)
  core/logging/RoutingLogEntry.kt     ← entidade Room
  core/logging/RoutingLogDao.kt       ← DAO para export
```

---

## Fase 1 — ServerInferenceService (~1 dia)

### 1.1 Dependencias (build.gradle.kts app)

```kotlin
// Retrofit para HTTP (se nao tiver)
implementation("com.squareup.retrofit2:retrofit:2.9.0")
implementation("com.squareup.retrofit2:converter-gson:2.9.0")
implementation("com.squareup.okhttp3:okhttp:4.12.0")
```

### 1.2 ServerConfig.kt

```kotlin
package com.voiceassistant.ai_server.model

data class ServerConfig(
    val baseUrl: String = "http://192.168.1.100:8080",
    val timeoutMs: Long = 30_000,
    val maxTokens: Int = 512,
    val temperature: Float = 0.7f,
    val topP: Float = 0.9f,
    val topK: Int = 40,
    val nProbs: Int = 5,  // top-k probs retornados pelo server
    val confidenceThresholdHigh: Float = 0.7f,
    val confidenceThresholdLow: Float = 0.3f
)
```

### 1.3 LlamaServerApi.kt (Retrofit interface)

```kotlin
package com.voiceassistant.ai_server.service

import retrofit2.http.Body
import retrofit2.http.POST

interface LlamaServerApi {
    @POST("/completion")
    suspend fun completion(@Body request: CompletionRequest): CompletionResponse
}

data class CompletionRequest(
    val prompt: String,
    val n_predict: Int = 512,
    val temperature: Float = 0.7f,
    val top_p: Float = 0.9f,
    val top_k: Int = 40,
    val n_probs: Int = 5  // ativa retorno de logprobs
)

data class CompletionResponse(
    val content: String,
    val tokens_predicted: Int,
    val completion_probabilities: List<TokenProb>? = null
)

data class TokenProb(
    val content: String,
    val probs: List<ProbEntry>
)

data class ProbEntry(
    val tok_str: String,
    val prob: Float
)
```

### 1.4 ServerInferenceService.kt

```kotlin
package com.voiceassistant.ai_server.service

import android.util.Log
import com.voiceassistant.ai_server.model.ServerConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ServerInferenceService @Inject constructor(
    private val config: ServerConfig
) {
    private var api: LlamaServerApi? = null

    val isAvailable: Boolean
        get() = api != null

    fun initialize() {
        val client = OkHttpClient.Builder()
            .connectTimeout(5, TimeUnit.SECONDS)
            .readTimeout(config.timeoutMs, TimeUnit.MILLISECONDS)
            .build()

        api = Retrofit.Builder()
            .baseUrl(config.baseUrl)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(LlamaServerApi::class.java)
    }

    /**
     * Gera resposta via llama-server e calcula confianca a partir dos logprobs.
     * Retorna par (texto, confianca). Confianca em [0,1].
     */
    suspend fun generateWithConfidence(prompt: String): ServerResult =
        withContext(Dispatchers.IO) {
            val serverApi = api ?: throw ServerUnavailableException("Servidor nao inicializado")

            val start = System.currentTimeMillis()

            val response = try {
                serverApi.completion(
                    CompletionRequest(
                        prompt = prompt,
                        n_predict = config.maxTokens,
                        temperature = config.temperature,
                        top_p = config.topP,
                        top_k = config.topK,
                        n_probs = config.nProbs
                    )
                )
            } catch (e: Exception) {
                throw ServerUnavailableException("Erro ao conectar: ${e.message}", e)
            }

            val latency = System.currentTimeMillis() - start
            val confidence = calculateConfidence(response.completion_probabilities)

            Log.i(TAG, "SERVER: ${response.tokens_predicted} tokens, " +
                    "confidence=${"%.3f".format(confidence)}, ${latency}ms")

            ServerResult(
                text = response.content,
                confidence = confidence,
                tokenCount = response.tokens_predicted,
                latencyMs = latency
            )
        }

    /**
     * Calcula confianca como media das probabilidades do token top-1
     * (o token efetivamente escolhido) em cada posicao.
     */
    private fun calculateConfidence(probs: List<TokenProb>?): Float {
        if (probs.isNullOrEmpty()) return -1f

        val tokenConfidences = probs.mapNotNull { tp ->
            // O primeiro prob entry e sempre o token escolhido
            tp.probs.firstOrNull()?.prob
        }

        if (tokenConfidences.isEmpty()) return -1f
        return tokenConfidences.average().toFloat()
    }

    companion object {
        private const val TAG = "LlamaServer"
    }
}

data class ServerResult(
    val text: String,
    val confidence: Float,  // [0,1] ou -1 se indisponivel
    val tokenCount: Int,
    val latencyMs: Long
)

class ServerUnavailableException(
    message: String, cause: Throwable? = null
) : Exception(message, cause)
```

---

## Fase 2 — Evoluir InferenceRouter (~1-2 dias)

### 2.1 Atualizar InferenceSource

```kotlin
enum class InferenceSource {
    LOCAL,
    SERVER,   // NOVO
    CLOUD,
    FALLBACK
}
```

### 2.2 Atualizar InferenceResult

```kotlin
data class InferenceResult(
    val text: String,
    val source: InferenceSource,
    val latencyMs: Long,
    val confidence: Float = -1f  // NOVO: -1 = nao disponivel
)
```

### 2.3 Atualizar RoutingDecision

```kotlin
enum class RoutingDecision {
    LOCAL,
    SERVER,                        // NOVO
    SERVER_WITH_CLOUD_ESCALATION,  // NOVO: server, mas escala para cloud se confianca baixa
    CLOUD,
    LOCAL_WITH_SERVER_FALLBACK,    // NOVO: local, fallback para server
    LOCAL_WITH_CLOUD_FALLBACK,
    ERROR_PRIVACY,
    ERROR_OFFLINE,
    ERROR_UNAVAILABLE;

    val targetsLocal: Boolean
        get() = this == LOCAL || this == LOCAL_WITH_SERVER_FALLBACK || this == LOCAL_WITH_CLOUD_FALLBACK
}
```

### 2.4 Nova logica de resolveRoute

```kotlin
companion object {
    fun resolveRoute(
        isOnline: Boolean,
        isLocalAvailable: Boolean,
        isServerAvailable: Boolean,  // NOVO
        isCloudAvailable: Boolean,
        complexity: PromptComplexity,
        privacyMode: Boolean
    ): RoutingDecision = when {
        // Privacidade: dados nunca saem do device
        privacyMode && isLocalAvailable -> RoutingDecision.LOCAL
        privacyMode -> RoutingDecision.ERROR_PRIVACY

        // Offline total (sem rede nenhuma)
        !isOnline && !isServerAvailable && isLocalAvailable -> RoutingDecision.LOCAL
        !isOnline && !isServerAvailable -> RoutingDecision.ERROR_OFFLINE

        // Complexa + cloud disponivel → cloud direto
        complexity == PromptComplexity.COMPLEX && isCloudAvailable -> RoutingDecision.CLOUD

        // Servidor disponivel (rede local) → usa servidor com logprobs
        isServerAvailable && isCloudAvailable ->
            RoutingDecision.SERVER_WITH_CLOUD_ESCALATION

        isServerAvailable ->
            RoutingDecision.SERVER

        // Sem servidor, mas local + cloud
        isLocalAvailable && isCloudAvailable ->
            RoutingDecision.LOCAL_WITH_CLOUD_FALLBACK

        isLocalAvailable ->
            RoutingDecision.LOCAL

        isCloudAvailable ->
            RoutingDecision.CLOUD

        else -> RoutingDecision.ERROR_UNAVAILABLE
    }
}
```

### 2.5 Execucao com confianca (pos-geracao)

```kotlin
private suspend fun runServerWithEscalation(prompt: String): InferenceResult {
    val result = serverService.generateWithConfidence(prompt)

    return when {
        result.confidence >= config.confidenceThresholdHigh -> {
            // Confianca alta → entrega diretamente
            InferenceResult(
                text = cleanResponse(result.text),
                source = InferenceSource.SERVER,
                latencyMs = result.latencyMs,
                confidence = result.confidence
            )
        }
        result.confidence < config.confidenceThresholdLow && cloudService.isAvailable -> {
            // Confianca baixa → escala para cloud
            Log.i(TAG, "Confianca baixa (${result.confidence}), escalonando para cloud")
            val cloudResult = runCloud(prompt)
            cloudResult.copy(confidence = result.confidence) // preserva confianca original para log
        }
        else -> {
            // Confianca media → entrega server com sinal de scaffolded
            InferenceResult(
                text = cleanResponse(result.text),
                source = InferenceSource.SERVER,
                latencyMs = result.latencyMs,
                confidence = result.confidence
            )
        }
    }
}
```

### 2.6 Adaptar modo pedagogico por confianca

No `infer()`, apos receber o resultado com confianca:

```kotlin
// Se confianca media e modo nao e HINT, adaptar para scaffolded
val effectiveMode = if (
    result.confidence in config.confidenceThresholdLow..config.confidenceThresholdHigh
    && request.tutorMode != TutorMode.HINT
) {
    TutorMode.HINT  // scaffolded-support
} else {
    request.tutorMode
}
```

**Nota**: Isso requer gerar a resposta pelo servidor e, se confianca media, re-gerar com prompt de HINT. Alternativa mais simples: sempre gerar com o modo solicitado e adicionar um aviso na UI ("O modelo tem confianca media nesta resposta — procure o professor se tiver duvidas").

---

## Fase 3 — Hilt + Config (~0.5 dia)

### 3.1 ServiceModule.kt atualizado

```kotlin
@Module
@InstallIn(SingletonComponent::class)
abstract class ServiceModule {

    @Binds @Singleton
    abstract fun bindLocalInference(impl: MediaPipeLocalInferenceService): LocalInferenceService

    @Binds @Singleton
    abstract fun bindCloudInference(impl: FirebaseCloudInferenceService): CloudInferenceService

    companion object {
        @Provides
        fun provideLocalModelConfig(): LocalModelConfig = LocalModelConfig()

        @Provides
        fun provideCloudModelConfig(): CloudModelConfig = CloudModelConfig()

        // NOVO
        @Provides
        @Singleton
        fun provideServerConfig(): ServerConfig = ServerConfig(
            baseUrl = "http://192.168.1.100:8080"  // configuravel via settings
        )
    }
}
```

### 3.2 InferenceRouter constructor atualizado

```kotlin
@Singleton
class InferenceRouter @Inject constructor(
    private val localService: LocalInferenceService,
    private val cloudService: CloudInferenceService,
    private val serverService: ServerInferenceService,  // NOVO
    private val localModelManager: LocalModelManager,
    private val networkMonitor: NetworkMonitor,
    private val userSettingsDataStore: UserSettingsDataStore,
    private val promptBuilder: TutorPromptBuilder,
    private val routingLogger: RoutingLogger  // NOVO
) : InferenceRepository {
```

---

## Fase 4 — Log de Pesquisa (~0.5 dia)

### 4.1 RoutingLogEntry.kt (Room entity)

```kotlin
package com.voiceassistant.core.logging

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "routing_log")
data class RoutingLogEntry(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val timestamp: Long = System.currentTimeMillis(),
    val sessionId: String,
    val questionText: String,
    val complexityPreFilter: String,   // SIMPLE/MODERATE/COMPLEX
    val routeDecision: String,         // RoutingDecision.name
    val confidenceScore: Float,        // -1 se indisponivel
    val confidenceMethod: String,      // "logprobs_mean" | "heuristic" | "none"
    val finalTier: String,             // InferenceSource.name
    val pedagogicalMode: String,       // TutorMode.name
    val latencyMs: Long,
    val modelId: String,               // "gemma-3-1b-Q4_K_M" | "firebase-gemini"
    val connectivity: String           // "offline" | "lan" | "internet"
)
```

### 4.2 RoutingLogDao.kt

```kotlin
@Dao
interface RoutingLogDao {
    @Insert
    suspend fun insert(entry: RoutingLogEntry)

    @Query("SELECT * FROM routing_log ORDER BY timestamp DESC")
    suspend fun getAll(): List<RoutingLogEntry>

    @Query("SELECT * FROM routing_log ORDER BY timestamp DESC LIMIT :limit")
    suspend fun getRecent(limit: Int = 100): List<RoutingLogEntry>
}
```

### 4.3 RoutingLogger.kt

```kotlin
@Singleton
class RoutingLogger @Inject constructor(
    private val dao: RoutingLogDao
) {
    suspend fun log(
        sessionId: String,
        questionText: String,
        complexity: PromptComplexity,
        decision: RoutingDecision,
        confidence: Float,
        confidenceMethod: String,
        finalSource: InferenceSource,
        mode: TutorMode,
        latencyMs: Long,
        modelId: String,
        connectivity: String
    ) {
        dao.insert(RoutingLogEntry(
            sessionId = sessionId,
            questionText = questionText,
            complexityPreFilter = complexity.name,
            routeDecision = decision.name,
            confidenceScore = confidence,
            confidenceMethod = confidenceMethod,
            finalTier = finalSource.name,
            pedagogicalMode = mode.name,
            latencyMs = latencyMs,
            modelId = modelId,
            connectivity = connectivity
        ))
    }

    suspend fun exportCsv(): String {
        val entries = dao.getAll()
        val header = "timestamp,session_id,question,complexity,route,confidence,method,tier,mode,latency_ms,model,connectivity"
        val rows = entries.joinToString("\n") { e ->
            "${e.timestamp},${e.sessionId},\"${e.questionText.take(100)}\",${e.complexityPreFilter}," +
            "${e.routeDecision},${e.confidenceScore},${e.confidenceMethod},${e.finalTier}," +
            "${e.pedagogicalMode},${e.latencyMs},${e.modelId},${e.connectivity}"
        }
        return "$header\n$rows"
    }
}
```

---

## Fase 5 — Setup do servidor llama.cpp

### Como rodar o servidor (fora do app — na maquina da escola/lab)

```bash
# Baixar llama.cpp pre-compilado ou compilar
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && make -j

# Baixar modelo GGUF
# Exemplo: Gemma 3 1B IT quantizado
wget https://huggingface.co/lmstudio-community/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf

# Iniciar servidor
./llama-server \
  -m gemma-3-1b-it-Q4_K_M.gguf \
  --port 8080 \
  --host 0.0.0.0 \
  -c 2048 \
  -t 4 \
  --n-probs 5
```

O servidor expoe:
- `GET /health` — verificar se esta rodando
- `POST /completion` — geracao com logprobs (quando `n_probs > 0`)

### Testar manualmente

```bash
curl http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "O que e fotossintese?", "n_predict": 100, "n_probs": 5}'
```

---

## Checklist de Validacao

- [ ] `llama-server` rodando e acessivel na rede local
- [ ] `ServerInferenceService.generateWithConfidence()` retorna texto + confidence
- [ ] Confidence em [0, 1] para respostas normais
- [ ] Confidence alta (~0.7+) para perguntas simples ("Quanto e 2+2?")
- [ ] Confidence baixa (~0.2-0.4) para perguntas muito fora do dominio
- [ ] `InferenceRouter` escalona para cloud quando confidence < threshold
- [ ] Tier local (MediaPipe) continua funcionando igual, sem regressao
- [ ] Modo privacidade bloqueia server e cloud (so usa local)
- [ ] Cenario offline: app funciona com local apenas
- [ ] Log de roteamento registra cada interacao
- [ ] Export CSV funciona

---

## Ordem de execucao

```
Dia 1:  Fase 1 (ServerInferenceService + API)
        - Retrofit setup
        - CompletionRequest/Response models
        - ServerInferenceService com calculateConfidence
        - Testar contra llama-server rodando local

Dia 2:  Fase 2 (InferenceRouter evoluido)
        - Atualizar InferenceSource, InferenceResult, RoutingDecision
        - Nova logica resolveRoute com isServerAvailable
        - runServerWithEscalation
        - Testar fluxo completo: servidor → confianca → decisao

Dia 3:  Fase 3 + 4 (Hilt + Log)
        - ServiceModule com ServerConfig
        - Room entity + DAO + Logger
        - Integrar log no InferenceRouter.infer()
        - Testar export CSV

Dia 4:  Testes + polish
        - Testes unitarios do novo resolveRoute
        - Teste E2E com servidor real
        - Cenarios: offline, lan-only, full internet
```

---

## Riscos e mitigacoes

| Risco | Mitigacao |
|---|---|
| Servidor cai durante uso | Fallback: InferenceRouter trata ServerUnavailableException como "servidor indisponivel" → usa local ou cloud |
| Latencia alta na rede local | Timeout de 5s no connect; 30s no read; se timeout → fallback |
| llama-server API muda entre versoes | Fixar versao; endpoint `/completion` e estavel ha 2+ anos |
| Modelo no servidor diferente do local | Registrar `modelId` no log; thresholds calibrados por modelo |

---

## O que NAO muda

- `MediaPipeLocalInferenceService` — intacto, tier offline
- `FirebaseCloudInferenceService` — intacto, tier cloud
- `ChatViewModel` — nao sabe do servidor (acessa via InferenceRepository)
- `SendMessageUseCase` — sem mudanca
- `TutorPromptBuilder` — sem mudanca (compact=true para server, mesma logica)
- `PromptComplexityAnalyzer` — sem mudanca, continua como pre-filtro rapido
- UI/Compose — sem mudanca (pode mostrar `InferenceSource.SERVER` no badge)
