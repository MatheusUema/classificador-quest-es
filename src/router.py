"""
Roteador de LLM baseado em dificuldade estimada da questão.

Score d ∈ [0, 1] dividido em três zonas pelos thresholds θ₁ e θ₂:

  [0, θ₁)    → local    (LLM leve no dispositivo)
  [θ₁, θ₂)  → servidor (LLM intermediário na rede local)
  [θ₂, 1]   → cloud    (LLM mais capaz, acesso remoto)

Os thresholds são configuráveis por escola / infraestrutura sem retreino.
"""

import sys
import numpy as np
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import extrair_features, FEATURES

BASE_DIR = Path(__file__).parent.parent
_modelo = None


def _carregar_modelo():
    """Carrega o classificador uma única vez (lazy loading)."""
    global _modelo
    if _modelo is None:
        path = BASE_DIR / "models" / "classificador.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em {path}.\n"
                "Execute: python src/train.py"
            )
        _modelo = joblib.load(path)
    return _modelo


def rotear(
    texto: str,
    area: str = "",
    theta1: float = 0.35,
    theta2: float = 0.65,
) -> dict:
    """
    Roteia uma questão para o LLM adequado.

    Args:
        texto:  Enunciado da questão.
        area:   Área do conhecimento (LC, CH, CN, MT). Opcional.
        theta1: Limiar inferior (abaixo → local). Padrão 0.35.
        theta2: Limiar superior (acima → cloud). Padrão 0.65.

    Returns:
        {"score": float, "destino": "local" | "servidor" | "cloud"}
    """
    modelo = _carregar_modelo()
    feats = extrair_features(texto, area)
    X = np.array([[feats.get(f, 0) for f in FEATURES]], dtype=np.float32)
    score = float(np.clip(modelo.predict(X)[0], 0.0, 1.0))

    if score < theta1:
        destino = "local"
    elif score < theta2:
        destino = "servidor"
    else:
        destino = "cloud"

    return {"score": round(score, 3), "destino": destino}


# ---------------------------------------------------------------------------
# Exemplos de teste
# ---------------------------------------------------------------------------
_EXEMPLOS = [
    # --- Simples (esperado: local) ---
    ("Quanto é 5 + 3?",
     "Aritmética básica", "local"),
    ("Quem descobriu o Brasil?",
     "Pergunta factual de história", "local"),
    ("Qual é a capital do Brasil?",
     "Geografia factual", "local"),

    # --- Médios (esperado: servidor) ---
    ("A Revolução Industrial iniciada na Inglaterra no século XVIII provocou "
     "mudanças profundas nas relações de trabalho. Quais foram as principais "
     "consequências sociais desse processo?",
     "História, análise de médio alcance", "servidor"),
    ("Em um triângulo retângulo, os catetos medem 3 cm e 4 cm. "
     "Qual é o comprimento da hipotenusa? Justifique.",
     "Matemática básica com justificativa", "servidor"),
    ("Explique o papel do Iluminismo na formação do pensamento político "
     "moderno e sua influência nas revoluções do século XVIII.",
     "Análise histórica", "servidor"),
    ("Um objeto de 2 kg é submetido a força resultante de 10 N. "
     "Calcule a aceleração e o espaço percorrido em 4 s.",
     "Física, aplicação de fórmulas", "servidor"),

    # --- Difíceis (esperado: cloud) ---
    ("Seja f(x) = x³ − 6x² + 9x − 4. Determine os intervalos em que f é "
     "crescente e decrescente, os extremos locais e os pontos de inflexão, "
     "utilizando o cálculo diferencial e justificando cada etapa.",
     "Cálculo diferencial completo", "cloud"),
    ("O processo de especiação simpátrica ocorre sem isolamento geográfico. "
     "Explique como a seleção natural leva ao isolamento reprodutivo em "
     "populações que coabitam o mesmo ambiente, citando mecanismos "
     "pré-zigóticos e pós-zigóticos.",
     "Biologia evolutiva avançada", "cloud"),
    ("A partir de Gadamer: 'A linguagem não é apenas instrumento de "
     "comunicação, mas o medium em que a realidade humana se constitui.' "
     "Discuta a relação entre linguagem, pensamento e realidade social, "
     "articulando com as teorias de Saussure e Wittgenstein.",
     "Filosofia da linguagem", "cloud"),
]


if __name__ == "__main__":
    print("Roteador de LLM — teste com 10 exemplos  (θ₁=0.35, θ₂=0.65)\n")
    cabecalho = f"{'Descrição':<38} {'Score':>6}  {'Destino':<10}  {'Esperado':<10}  OK?"
    print(cabecalho)
    print("-" * len(cabecalho))

    acertos = 0
    for texto, descricao, esperado in _EXEMPLOS:
        r = rotear(texto)
        ok = "✓" if r["destino"] == esperado else "✗"
        if r["destino"] == esperado:
            acertos += 1
        print(f"{descricao:<38} {r['score']:>6.3f}  {r['destino']:<10}  {esperado:<10}  {ok}")

    print(f"\nAcertos: {acertos}/{len(_EXEMPLOS)}")
