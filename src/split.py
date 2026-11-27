"""Divisao estratificada 70/15/15 com semente fixa, gerada uma unica vez.

Uso:  python -m src.split            (gera results/split.csv)
      python -m src.split --verificar (so confere o arquivo existente)
"""

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CLASSES, DATA_DIR, PROPORCOES, SEED, SPLIT_CSV
from src.dataset import listar_imagens

ORDEM_CONJUNTOS = ["treino", "validacao", "teste"]


def gerar_split(df, seed=SEED):
    """Duas divisoes estratificadas: 70 / 30 e depois 30 -> 15 / 15."""
    frac_resto = PROPORCOES["validacao"] + PROPORCOES["teste"]
    treino, resto = train_test_split(
        df, test_size=frac_resto, stratify=df["classe"], random_state=seed
    )
    frac_teste = PROPORCOES["teste"] / frac_resto
    validacao, teste = train_test_split(
        resto, test_size=frac_teste, stratify=resto["classe"], random_state=seed
    )
    partes = []
    for nome, parte in zip(ORDEM_CONJUNTOS, [treino, validacao, teste]):
        partes.append(parte.assign(conjunto=nome))
    out = pd.concat(partes)
    out["conjunto"] = pd.Categorical(out["conjunto"], ORDEM_CONJUNTOS, ordered=True)
    out["classe"] = pd.Categorical(out["classe"], CLASSES, ordered=True)
    out = out.sort_values(["conjunto", "classe", "caminho"]).reset_index(drop=True)
    out["conjunto"] = out["conjunto"].astype(str)
    out["classe"] = out["classe"].astype(str)
    return out[["caminho", "classe", "conjunto"]]


def carregar_split(caminho=SPLIT_CSV):
    return pd.read_csv(caminho)


def tabela_contagens(split):
    """Classes nas linhas, conjuntos nas colunas."""
    tab = pd.crosstab(split["classe"], split["conjunto"])
    return tab.reindex(index=CLASSES, columns=ORDEM_CONJUNTOS)


def verificar_split(split):
    """Falha se houver vazamento ou se a divisao fugir do esperado."""
    assert split["caminho"].is_unique, "imagem repetida no split"
    conjuntos = {c: set(split.loc[split["conjunto"] == c, "caminho"]) for c in ORDEM_CONJUNTOS}
    for i, a in enumerate(ORDEM_CONJUNTOS):
        for b in ORDEM_CONJUNTOS[i + 1:]:
            assert not conjuntos[a] & conjuntos[b], f"vazamento entre {a} e {b}"
    assert set(split["classe"]) == set(CLASSES), "faltam classes"
    tab = tabela_contagens(split)
    assert (tab > 0).all().all(), "classe sem exemplo em algum conjunto"
    fracs = split["conjunto"].value_counts(normalize=True)
    for nome, esperado in PROPORCOES.items():
        assert abs(fracs[nome] - esperado) < 0.01, f"proporcao de {nome} fora do esperado"
    return tab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--verificar", action="store_true")
    args = parser.parse_args()

    if args.verificar:
        split = carregar_split()
    else:
        split = gerar_split(listar_imagens(args.data_dir))
        SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(SPLIT_CSV, index=False)
        print(f"Salvo em {SPLIT_CSV}")

    tab = verificar_split(split)
    print(f"Total: {len(split)} imagens, semente {SEED}")
    print(tab.assign(total=tab.sum(axis=1)).to_string())
    print("Verificacao OK: sem vazamento, estratificado.")


if __name__ == "__main__":
    main()
