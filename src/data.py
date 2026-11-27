"""Carregamento das imagens por resolucao e pesos de classe.

Decisoes:
- Toda imagem e lida da versao 512x384 do dataset-resized.zip, convertida
  para RGB e redimensionada com PIL (filtro bilinear, com antialias na
  reducao). Em 512x384 nao ha redimensionamento.
- As imagens ficam em memoria como uint8 (0-255). A normalizacao /255 e o
  aumento de dados ficam dentro do modelo (src/models.py): o aumento so atua
  em model.fit (treino) e fica desligado em validacao, teste e previsao.
- Os pesos de classe usam apenas as contagens do conjunto de treino.
"""

import numpy as np
from PIL import Image

from src.config import CLASSES, DATA_DIR, RESOLUCAO_ORIGINAL
from src.dataset import encontrar_raiz

INDICE_CLASSE = {c: i for i, c in enumerate(CLASSES)}


def ler_imagem(caminho, resolucao):
    """Le um arquivo e devolve array uint8 (altura, largura, 3)."""
    with Image.open(caminho) as img:
        img = img.convert("RGB")
        if img.size != RESOLUCAO_ORIGINAL:
            raise ValueError(f"{caminho}: esperado {RESOLUCAO_ORIGINAL}, veio {img.size}")
        if tuple(resolucao) != RESOLUCAO_ORIGINAL:
            img = img.resize(tuple(resolucao), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def carregar_conjunto(split, conjunto, resolucao, data_dir=DATA_DIR):
    """Imagens e rotulos (indices em CLASSES) de um conjunto do split.

    Retorna X uint8 (n, altura, largura, 3), y int32 (n,) e a lista de caminhos,
    na ordem do split.csv.
    """
    raiz = encontrar_raiz(data_dir)
    parte = split[split["conjunto"] == conjunto]
    caminhos = parte["caminho"].tolist()
    X = np.stack([ler_imagem(raiz / c, resolucao) for c in caminhos])
    y = parte["classe"].map(INDICE_CLASSE).to_numpy(dtype=np.int32)
    return X, y, caminhos


def pesos_de_classe(split):
    """Peso da classe c = n_treino / (n_classes * n_treino_c).

    Inversamente proporcional a frequencia; a media ponderada dos pesos no
    treino e 1, entao a escala da perda nao muda.
    """
    treino = split[split["conjunto"] == "treino"]
    contagens = treino["classe"].value_counts()
    n = len(treino)
    return {INDICE_CLASSE[c]: float(n / (len(CLASSES) * contagens[c])) for c in CLASSES}
