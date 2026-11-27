"""Estilo comum das figuras (EDA e resultados)."""

from pathlib import Path

import matplotlib.pyplot as plt

from src.config import RESULTS_DIR

FIG_DIR = RESULTS_DIR / "figures"

# Paleta categorica validada (ordem fixa) e tons de texto/superficie.
AZUL, LARANJA, VERDE_AGUA, AMARELO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TEXTO, TEXTO_2, GRADE, FUNDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COR_CONJUNTO = {"treino": AZUL, "validacao": LARANJA, "teste": VERDE_AGUA}
ROTULO_CONJUNTO = {"treino": "Treino", "validacao": "Validação", "teste": "Teste"}


def aplicar_estilo():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.facecolor": FUNDO,
        "axes.facecolor": FUNDO,
        "axes.edgecolor": GRADE,
        "axes.labelcolor": TEXTO_2,
        "axes.titlecolor": TEXTO,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRADE,
        "grid.linewidth": 0.8,
        "xtick.color": TEXTO_2,
        "ytick.color": TEXTO_2,
        "legend.frameon": False,
        "font.size": 9,
    })


def salvar(fig, nome):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    caminho = Path(FIG_DIR) / nome
    fig.savefig(caminho)
    return caminho
