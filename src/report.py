"""Leitura das saidas dos experimentos e exportacao dos arquivos leves."""

import json
import zipfile
from pathlib import Path

import pandas as pd

from src.config import ARQUIVOS_LEVES, EXPERIMENTOS, RESULTS_EXP_DIR


def carregar_experimento(nome, pasta=RESULTS_EXP_DIR):
    """Dicionario com config, custo, historico e metricas de um experimento."""
    base = Path(pasta) / nome
    out = {}
    for arq in ["config", "cost", "history", "metrics"]:
        caminho = base / f"{arq}.json"
        out[arq] = json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else None
    prev = base / "predictions.csv"
    out["predictions"] = pd.read_csv(prev) if prev.exists() else None
    return out


def concluidos(pasta=RESULTS_EXP_DIR):
    return [n for n in EXPERIMENTOS if (Path(pasta) / n / "metrics.json").exists()]


def tabela_resumo(pasta=RESULTS_EXP_DIR):
    """Uma linha por experimento: metricas de validacao/teste e custo."""
    linhas = []
    for nome in concluidos(pasta):
        r = carregar_experimento(nome, pasta)
        exp, c, m = EXPERIMENTOS[nome], r["cost"], r["metrics"]
        linhas.append({
            "experimento": nome,
            "modelo": exp["modelo"].upper(),
            "resolucao": f"{exp['resolucao'][0]}x{exp['resolucao'][1]}",
            "pesos_classe": "sim" if exp["pesos_classe"] else "nao",
            "f1_macro_val": m["validacao"]["f1_macro"],
            "acuracia_teste": m["teste"]["acuracia"],
            "f1_macro_teste": m["teste"]["f1_macro"],
            "parametros": c["parametros"],
            "tamanho_mb": c["tamanho_modelo_mb"],
            "epocas": c["epocas_executadas"],
            "epoca_restaurada": c["epoca_restaurada"],
            "tempo_total_s": c["tempo_total_s"],
            "tempo_epoca_s": c["tempo_medio_epoca_s"],
            "pico_vram_mb": c["pico_vram_mb"],
            "pico_ram_mb": c["pico_ram_processo_mb"],
            "hardware": c["hardware"],
            "lote": c["batch_size"],
        })
    return pd.DataFrame(linhas).set_index("experimento") if linhas else pd.DataFrame()


def exportar_leves(outputs_dir, destino_zip):
    """Zip com experimentos/<nome>/<arquivos leves> (sem model.keras).

    Descompactar dentro de results/ no repositorio.
    """
    destino_zip = Path(destino_zip)
    destino_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destino_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for nome in concluidos(outputs_dir):
            for arq in ARQUIVOS_LEVES:
                origem = Path(outputs_dir) / nome / arq
                if origem.exists():
                    z.write(origem, f"experimentos/{nome}/{arq}")
    return destino_zip
