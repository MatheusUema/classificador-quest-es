"""
Treino, avaliação e exportação do classificador de dificuldade de questões ENEM.
Gera classificador.joblib (Python/servidor) e classificador.onnx (mobile).
"""

import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import FEATURES, PROCESSED_DIR

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "difficulty_score"


def treinar(
    dataset_path: Path = PROCESSED_DIR / "dataset_enem_dificuldade.csv",
) -> GradientBoostingRegressor:
    """
    Carrega o dataset, treina GradientBoostingRegressor e avalia no teste.
    Split 80/20 estratificado por área para garantir representação balanceada.
    """
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)
    areas = df["area"].values  # usado para stratify

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=areas
    )
    print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

    modelo = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nMAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")

    _plotar_importancia(FEATURES, modelo.feature_importances_)
    return modelo


def salvar_joblib(modelo: GradientBoostingRegressor) -> Path:
    """Serializa o modelo para uso em Python / servidor."""
    path = MODELS_DIR / "classificador.joblib"
    joblib.dump(modelo, path)
    print(f"\nModelo salvo : {path}")
    return path


def exportar_onnx(modelo: GradientBoostingRegressor) -> "Path | None":
    """
    Converte o modelo para ONNX e valida a equivalência numérica.
    Requer skl2onnx e onnxruntime instalados.
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnxruntime as rt
    except ImportError as e:
        print(f"[AVISO] Exportação ONNX pulada — dependência ausente: {e}")
        return None

    n_feat = len(FEATURES)
    onnx_model = convert_sklearn(
        modelo,
        initial_types=[("float_input", FloatTensorType([None, n_feat]))],
    )
    path = MODELS_DIR / "classificador.onnx"
    path.write_bytes(onnx_model.SerializeToString())
    print(f"Modelo ONNX salvo : {path}")

    # Validação: compara predições joblib × ONNX em 100 amostras aleatórias
    rng = np.random.default_rng(0)
    X_val = rng.random((100, n_feat)).astype(np.float32)
    pred_sk = modelo.predict(X_val)

    sess = rt.InferenceSession(str(path))
    pred_onnx = sess.run(None, {sess.get_inputs()[0].name: X_val})[0].flatten()

    max_diff = float(np.abs(pred_sk - pred_onnx).max())
    print(f"Validação ONNX — diferença máxima vs joblib: {max_diff:.2e}")
    if max_diff > 1e-3:
        print("[AVISO] Diferença acima de 1e-3 — verifique a versão do skl2onnx.")

    return path


def _plotar_importancia(features: list, importancias: np.ndarray) -> None:
    ordem = np.argsort(importancias)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([features[i] for i in ordem], importancias[ordem])
    ax.set_title("Feature Importance — Classificador de Dificuldade")
    ax.set_xlabel("Importância")
    plt.tight_layout()
    out = PROCESSED_DIR / "feature_importance.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Feature importance : {out}")


if __name__ == "__main__":
    modelo = treinar()
    salvar_joblib(modelo)
    exportar_onnx(modelo)
