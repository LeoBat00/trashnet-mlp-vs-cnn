"""Metricas, matriz de confusao e previsoes.

Protocolo: o treino ajusta os pesos, a validacao escolhe (F1 macro de
validacao) e o teste so e usado no fim, sem nenhum reajuste depois.

zero_division=0: se um modelo nunca preve uma classe, a precisao dela e 0/0.
Com zero_division=0 esse valor vira 0 (em vez de aviso/NaN), o que penaliza o
modelo no F1 macro em vez de esconder o problema.

Avaliacao sem novo treino (usa os modelos salvos):
    python -m src.evaluate                 # todos os experimentos
    python -m src.evaluate --exp exp3_cnn_128x96
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from src.config import CLASSES, DATA_DIR, EXPERIMENTOS, OUTPUTS_DIR

ROTULOS = list(range(len(CLASSES)))


def metricas(y_real, y_prev):
    """Acuracia, F1 macro, relatorio por classe e matriz (reais x previstas)."""
    return {
        "acuracia": float(accuracy_score(y_real, y_prev)),
        "f1_macro": float(f1_score(y_real, y_prev, labels=ROTULOS, average="macro", zero_division=0)),
        "por_classe": classification_report(
            y_real, y_prev, labels=ROTULOS, target_names=CLASSES,
            output_dict=True, zero_division=0,
        ),
        # linhas = classes reais, colunas = previstas, na ordem de CLASSES
        "matriz_confusao": confusion_matrix(y_real, y_prev, labels=ROTULOS).tolist(),
    }


def prever(modelo, X, batch_size=32):
    probs = modelo.predict(X, batch_size=batch_size, verbose=0)
    return probs, probs.argmax(axis=1)


def tabela_previsoes(caminhos, y_real, probs):
    """Uma linha por imagem de teste, com a probabilidade de cada classe."""
    y_prev = probs.argmax(axis=1)
    df = pd.DataFrame({
        "caminho": caminhos,
        "real": [CLASSES[i] for i in y_real],
        "prevista": [CLASSES[i] for i in y_prev],
        "prob_prevista": probs.max(axis=1),
        "acertou": y_real == y_prev,
    })
    for i, c in enumerate(CLASSES):
        df[f"prob_{c}"] = probs[:, i]
    return df


def avaliar_salvo(nome, outputs_dir=OUTPUTS_DIR, data_dir=DATA_DIR):
    """Recarrega o modelo salvo e recalcula as metricas de teste."""
    from tensorflow import keras

    from src.data import carregar_conjunto
    from src.split import carregar_split

    pasta = Path(outputs_dir) / nome
    modelo = keras.models.load_model(pasta / "model.keras")
    X, y, _ = carregar_conjunto(carregar_split(), "teste", EXPERIMENTOS[nome]["resolucao"], data_dir)
    _, y_prev = prever(modelo, X)
    novo = metricas(y, y_prev)
    salvo = json.loads((pasta / "metrics.json").read_text())["teste"]
    return novo, salvo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all")
    parser.add_argument("--outputs-dir", default=OUTPUTS_DIR)
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()

    nomes = list(EXPERIMENTOS) if args.exp == "all" else [args.exp]
    for nome in nomes:
        novo, salvo = avaliar_salvo(nome, args.outputs_dir, args.data_dir)
        igual = np.isclose(novo["f1_macro"], salvo["f1_macro"]) and np.isclose(novo["acuracia"], salvo["acuracia"])
        print(f"{nome}: acuracia={novo['acuracia']:.4f}  f1_macro={novo['f1_macro']:.4f}  "
              f"{'igual ao salvo' if igual else 'DIFERENTE do salvo'}")


if __name__ == "__main__":
    main()
