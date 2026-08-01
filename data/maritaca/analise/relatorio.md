# Analise — Confianca do modelo local vs Acerto (ENEM/Maritaca)

Base: `resultados_acerto_local.csv` — **389** questoes avaliadas (texto puro, com IRT). Acuracia local global: **37.8%**.

## 1. Metricas por area

| Area | n | Acuracia | Conf. acerto | Conf. erro | AUC |
|---|---:|---:|---:|---:|---:|
| GLOBAL | 389 | 37.8% | 0.839 | 0.751 | 0.661 |
| LC | 112 | 48.2% | 0.860 | 0.802 | 0.634 |
| CH | 117 | 52.1% | 0.846 | 0.748 | 0.691 |
| CN | 77 | 22.1% | 0.838 | 0.757 | 0.638 |
| MT | 83 | 18.1% | 0.736 | 0.705 | 0.531 |

> CN/MT ficam proximas do acaso (20%) e a AUC nelas tende a ~0.5 — a confianca **nao** separa acerto de erro nessas areas; em LC/CH ha mais sinal.

![conf](./fig_conf_dist.png)

![roc](./fig_roc.png)

![acc](./fig_acc_area.png)

## 2. Calibracao (superconfianca)

**ECE = 0.406.** O modelo e superconfiante: diz ~0.8+ mas acerta bem menos (a curva fica abaixo da diagonal).

| bin conf | n | conf media | acuracia obs |
|---|---:|---:|---:|
| 0.2-0.3 | 2 | 0.262 | 0.000 |
| 0.3-0.4 | 13 | 0.336 | 0.308 |
| 0.4-0.5 | 36 | 0.450 | 0.278 |
| 0.5-0.6 | 41 | 0.552 | 0.341 |
| 0.6-0.7 | 40 | 0.654 | 0.200 |
| 0.7-0.8 | 37 | 0.746 | 0.297 |
| 0.8-0.9 | 50 | 0.856 | 0.260 |
| 0.9-1.0 | 170 | 0.969 | 0.512 |

![calib](./fig_calibracao.png)

## 3. Simulacao de politicas de roteamento

Proxy: **escalar = acertar** com probabilidade `p_esc` (o tier superior e forte). Comparamos acuracia final x taxa de escalonamento (custo).

![policy](./fig_policy_tradeoff.png)


### p_esc = 1.0

| Orcamento escal. | C (threshold) | D (area-aware) | E (pre-filtro IRT) |
|---|---:|---:|---:|
| ≤20% | 50.9% @ 19% | — | 53.2% @ 20% |
| ≤40% | 66.8% @ 40% | — | 67.4% @ 40% |
| ≤60% | 81.0% @ 60% | 82.0% @ 60% | 77.6% @ 58% |
| ≤80% | 93.6% @ 79% | 94.9% @ 80% | 89.7% @ 80% |

**Vencedor:** em ≤60% de escalonamento, **politica D** vence (82.0% @ 60%). [C=81.0%, D=82.0%, E=77.6%]


### p_esc = 0.85

| Orcamento escal. | C (threshold) | D (area-aware) | E (pre-filtro IRT) |
|---|---:|---:|---:|
| ≤20% | 48.0% @ 19% | — | 50.3% @ 20% |
| ≤40% | 60.9% @ 40% | — | 61.4% @ 40% |
| ≤60% | 72.0% @ 60% | 73.1% @ 60% | 69.1% @ 57% |
| ≤80% | 81.7% @ 79% | 82.9% @ 80% | 77.9% @ 79% |

**Vencedor:** em ≤60% de escalonamento, **politica D** vence (73.1% @ 60%). [C=72.0%, D=73.1%, E=69.1%]

## 4. Implicacoes para o app

- **Rota por area:** CN e MT devem ser **sempre escaladas** (local ~ acaso); o ganho por escalar essas areas primeiro e o maior. A confianca so ajuda a decidir dentro de **LC/CH**.

- **Threshold nao deve ser global e alto:** setar `confidenceThresholdHigh=0.98` (o Youden global) manteria local so os quase-certos e escalaria a grande maioria — caro. O corte otimo **depende do orcamento** de escalonamento.

- **Calibracao:** por causa da superconfianca (ECE alto), o valor bruto de confianca nao e uma probabilidade de acerto; use-o como *ranking* (via threshold calibrado por area), nao como probabilidade absoluta.

- **IRT confirmado como proxy fraco:** ver correlacoes de Spearman ~ -0.2 no run anterior; a politica E (pre-filtro IRT) fica atras de C/D.
