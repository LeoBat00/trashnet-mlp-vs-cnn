"""Listagem das imagens do dataset-resized.zip (TrashNet)."""

from pathlib import Path

import pandas as pd
from PIL import Image

from src.config import CLASSES, DATA_DIR

EXTENSOES = {".jpg", ".jpeg", ".png"}


def encontrar_raiz(data_dir=DATA_DIR):
    """Acha a pasta que contem as 6 pastas de classe (o zip pode ter subpasta)."""
    data_dir = Path(data_dir)
    candidatos = [data_dir, *[p for p in data_dir.rglob("*") if p.is_dir()]]
    for pasta in candidatos:
        if "__MACOSX" in pasta.parts:
            continue
        if all((pasta / classe).is_dir() for classe in CLASSES):
            return pasta
    raise FileNotFoundError(f"Pastas das classes {CLASSES} nao encontradas em {data_dir}")


def listar_imagens(data_dir=DATA_DIR):
    """Uma linha por imagem: caminho relativo, classe, dimensoes e modo de cor.

    Le so o cabecalho de cada arquivo (rapido). A verificacao completa de
    integridade fica no notebook de EDA.
    """
    raiz = encontrar_raiz(data_dir)
    linhas = []
    for classe in CLASSES:
        for arquivo in sorted((raiz / classe).iterdir()):
            if arquivo.suffix.lower() not in EXTENSOES or arquivo.name.startswith("."):
                continue
            with Image.open(arquivo) as img:
                largura, altura = img.size
                modo = img.mode
            linhas.append({
                "caminho": f"{classe}/{arquivo.name}",
                "classe": classe,
                "largura": largura,
                "altura": altura,
                "modo": modo,
            })
    return pd.DataFrame(linhas)
