"""Configuracao central: tudo que precisa ser igual entre os experimentos fica aqui."""

import os
from pathlib import Path

# --- Caminhos -------------------------------------------------------------
# Podem ser trocados por variavel de ambiente (no Colab: disco local e Drive).
RAIZ_REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("TRASHNET_DATA_DIR", RAIZ_REPO / "data"))
OUTPUTS_DIR = Path(os.environ.get("TRASHNET_OUTPUTS_DIR", RAIZ_REPO / "outputs"))
RESULTS_DIR = RAIZ_REPO / "results"
SPLIT_CSV = RESULTS_DIR / "split.csv"

# --- Dados ----------------------------------------------------------------
SEED = 42
# Ordem fixa: usada nos rotulos, nas metricas e nas matrizes de confusao.
CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
NUM_CLASSES = len(CLASSES)
PROPORCOES = {"treino": 0.70, "validacao": 0.15, "teste": 0.15}
RESOLUCAO_ORIGINAL = (512, 384)  # (largura, altura)

# --- Treino ---------------------------------------------------------------
# Lote, epocas e parada iguais para todos. A taxa de aprendizado e fixa por
# tipo de modelo, portanto igual dentro de cada comparacao controlada
# (com/sem pesos e as tres resolucoes da CNN).
BATCH_SIZE = 32
MAX_EPOCAS = 60
PACIENCIA = 10
LR = {"mlp": 1e-4, "cnn": 1e-3}

# --- Experimentos ---------------------------------------------------------
# resolucao = (largura, altura). No Keras a entrada e (altura, largura, 3).
EXPERIMENTOS = {
    "exp1_mlp_128x96": {"modelo": "mlp", "resolucao": (128, 96), "pesos_classe": False},
    "exp2_mlp_128x96_pesos": {"modelo": "mlp", "resolucao": (128, 96), "pesos_classe": True},
    "exp3_cnn_128x96": {"modelo": "cnn", "resolucao": (128, 96), "pesos_classe": False},
    "exp4_cnn_128x96_pesos": {"modelo": "cnn", "resolucao": (128, 96), "pesos_classe": True},
    "exp5_cnn_256x192": {"modelo": "cnn", "resolucao": (256, 192), "pesos_classe": False},
    "exp6_cnn_512x384": {"modelo": "cnn", "resolucao": (512, 384), "pesos_classe": False},
}


def input_shape(resolucao):
    """(largura, altura) -> formato de entrada do Keras (altura, largura, 3)."""
    largura, altura = resolucao
    return (altura, largura, 3)
