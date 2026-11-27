"""Treino de um experimento, com medicao de custo e retomada.

Uso:
    python -m src.train --exp exp3_cnn_128x96
    python -m src.train --exp all
    python -m src.train --exp all --max-epocas 1 --amostra 5 --outputs-dir /tmp/smoke   (teste rapido)

Saidas em <outputs>/<experimento>/:
    model.keras      modelo com os pesos da melhor epoca (menor val_loss)
    history.json     perda/acuracia por epoca (treino e validacao) + tempo de cada epoca
    config.json      hiperparametros, semente, pesos de classe, versoes, hardware
    cost.json        parametros, tamanho do modelo, epocas, tempos, picos de memoria
    predictions.csv  previsoes do teste
    metrics.json     metricas de validacao e teste (escrito por ultimo = experimento concluido)
"""

import argparse
import gc
import json
import os
import platform
import threading
import time
from pathlib import Path

import numpy as np
import psutil

from src.config import (BATCH_SIZE, CLASSES, DATA_DIR, EXPERIMENTOS, LR, MAX_EPOCAS,
                        OUTPUTS_DIR, PACIENCIA, SEED, input_shape)


# --------------------------------------------------------------------------
# Medicao
# --------------------------------------------------------------------------
class MedidorRAM:
    """Amostra o uso de RAM do processo (RSS) durante o treino, a cada 0,5 s."""

    def __init__(self, intervalo=0.5):
        self.intervalo = intervalo
        self.pico = 0
        self._parar = threading.Event()
        self._proc = psutil.Process()

    def _loop(self):
        while not self._parar.is_set():
            self.pico = max(self.pico, self._proc.memory_info().rss)
            time.sleep(self.intervalo)

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._parar.set()
        self._thread.join()
        self.pico = max(self.pico, self._proc.memory_info().rss)


def _callback_tempo():
    from tensorflow import keras

    class TempoPorEpoca(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.tempos = []

        def on_epoch_begin(self, epoch, logs=None):
            self._t0 = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            self.tempos.append(time.perf_counter() - self._t0)

    return TempoPorEpoca()


def descrever_hardware():
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        detalhes = tf.config.experimental.get_device_details(gpus[0])
        return {"tipo": "GPU", "nome": detalhes.get("device_name", "GPU desconhecida")}
    return {"tipo": "CPU", "nome": platform.processor() or platform.machine()}


def versoes():
    import keras
    import pandas
    import sklearn
    import tensorflow as tf

    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
    }


# --------------------------------------------------------------------------
# Treino
# --------------------------------------------------------------------------
def preparar_reprodutibilidade():
    """Mesma semente para Python, NumPy e TensorFlow + operacoes deterministicas."""
    import tensorflow as tf
    from tensorflow import keras

    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()


def _salvar_json(caminho, obj):
    Path(caminho).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def rodar_experimento(nome, split, outputs_dir=OUTPUTS_DIR, data_dir=DATA_DIR,
                      max_epocas=MAX_EPOCAS, refazer=False):
    import tensorflow as tf
    from tensorflow import keras

    from src.data import carregar_conjunto, pesos_de_classe
    from src.evaluate import metricas, prever, tabela_previsoes
    from src.models import criar_modelo

    exp = EXPERIMENTOS[nome]
    pasta = Path(outputs_dir) / nome
    if (pasta / "metrics.json").exists() and not refazer:
        print(f"[{nome}] ja concluido, pulando.")
        return json.loads((pasta / "metrics.json").read_text())
    pasta.mkdir(parents=True, exist_ok=True)

    # Dados (preparo fora da medicao de tempo)
    res = exp["resolucao"]
    X_tr, y_tr, _ = carregar_conjunto(split, "treino", res, data_dir)
    X_va, y_va, _ = carregar_conjunto(split, "validacao", res, data_dir)
    pesos = pesos_de_classe(split) if exp["pesos_classe"] else None

    # Modelo novo, mesma semente em todos os experimentos
    keras.backend.clear_session()
    preparar_reprodutibilidade()
    modelo = criar_modelo(exp["modelo"], input_shape(res))

    hardware = descrever_hardware()
    config = {
        "experimento": nome,
        **exp,
        "input_shape_keras": list(input_shape(res)),
        "semente": SEED,
        "batch_size": BATCH_SIZE,
        "otimizador": "Adam",
        "learning_rate": LR[exp["modelo"]],
        "perda": "sparse_categorical_crossentropy",
        "max_epocas": max_epocas,
        "early_stopping": {"monitor": "val_loss", "patience": PACIENCIA, "restore_best_weights": True},
        "pesos_de_classe": {CLASSES[i]: p for i, p in pesos.items()} if pesos else None,
        "n_treino": int(len(y_tr)),
        "n_validacao": int(len(y_va)),
        "hardware": hardware,
        "versoes": versoes(),
    }
    _salvar_json(pasta / "config.json", config)

    tempo_cb = _callback_tempo()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=PACIENCIA,
                                      restore_best_weights=True, verbose=1),
        # garante os pesos da melhor epoca mesmo se o treino chegar a max_epocas
        keras.callbacks.ModelCheckpoint(str(pasta / "melhor.weights.h5"), monitor="val_loss",
                                        save_best_only=True, save_weights_only=True),
        tempo_cb,
    ]

    usa_gpu = hardware["tipo"] == "GPU"
    if usa_gpu:
        tf.config.experimental.reset_memory_stats("GPU:0")

    # Tempo medido apenas em model.fit (sem download, descompactacao ou leitura das imagens)
    with MedidorRAM() as ram:
        t0 = time.perf_counter()
        hist = modelo.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=max_epocas,
            batch_size=BATCH_SIZE,
            class_weight=pesos,
            shuffle=True,
            callbacks=callbacks,
            verbose=2,
        )
        tempo_total = time.perf_counter() - t0

    pico_vram = tf.config.experimental.get_memory_info("GPU:0")["peak"] if usa_gpu else None

    modelo.load_weights(pasta / "melhor.weights.h5")
    (pasta / "melhor.weights.h5").unlink()
    modelo.save(pasta / "model.keras")

    historico = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    historico["tempo_epoca_s"] = tempo_cb.tempos
    melhor_epoca = int(np.argmin(historico["val_loss"])) + 1
    historico["melhor_epoca"] = melhor_epoca
    _salvar_json(pasta / "history.json", historico)

    custo = {
        "parametros": int(modelo.count_params()),
        "tamanho_modelo_mb": (pasta / "model.keras").stat().st_size / 2**20,
        "epocas_executadas": len(tempo_cb.tempos),
        "epoca_restaurada": melhor_epoca,
        "tempo_total_s": tempo_total,
        "tempo_medio_epoca_s": float(np.mean(tempo_cb.tempos)),
        "batch_size": BATCH_SIZE,
        "hardware": hardware["nome"],
        "pico_vram_mb": pico_vram / 2**20 if pico_vram is not None else None,
        "pico_ram_processo_mb": ram.pico / 2**20,
        "metodo_medicao": (
            "time.perf_counter em volta de model.fit (sem download, descompactacao e leitura "
            "das imagens, feitos antes); tempo por epoca via callback; pico de VRAM por "
            "tf.config.experimental.get_memory_info apos reset_memory_stats; pico de RAM do "
            "processo (RSS) amostrado a cada 0,5 s com psutil durante o fit"
        ),
    }
    _salvar_json(pasta / "cost.json", custo)

    # Validacao escolhe; teste so agora, sem reajuste
    _, y_va_prev = prever(modelo, X_va)
    del X_tr, X_va
    gc.collect()
    X_te, y_te, cam_te = carregar_conjunto(split, "teste", res, data_dir)
    probs_te, y_te_prev = prever(modelo, X_te)
    tabela_previsoes(cam_te, y_te, probs_te).to_csv(pasta / "predictions.csv", index=False)

    resultado = {
        "experimento": nome,
        "validacao": metricas(y_va, y_va_prev),
        "teste": metricas(y_te, y_te_prev),
    }
    _salvar_json(pasta / "metrics.json", resultado)
    print(f"[{nome}] val f1_macro={resultado['validacao']['f1_macro']:.4f} | "
          f"teste acuracia={resultado['teste']['acuracia']:.4f} "
          f"f1_macro={resultado['teste']['f1_macro']:.4f} | "
          f"{custo['epocas_executadas']} epocas, melhor {melhor_epoca}, {tempo_total:.1f}s")

    del modelo, X_te
    keras.backend.clear_session()
    gc.collect()
    return resultado


def main():
    from src.split import carregar_split

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all", help="nome do experimento ou 'all'")
    parser.add_argument("--outputs-dir", default=OUTPUTS_DIR)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--max-epocas", type=int, default=MAX_EPOCAS)
    parser.add_argument("--amostra", type=int, default=None,
                        help="so para teste rapido: n imagens por classe e conjunto")
    parser.add_argument("--refazer", action="store_true")
    args = parser.parse_args()

    split = carregar_split()
    if args.amostra:
        split = split.groupby(["conjunto", "classe"]).head(args.amostra).reset_index(drop=True)

    nomes = list(EXPERIMENTOS) if args.exp == "all" else args.exp.split(",")
    for nome in nomes:
        rodar_experimento(nome, split, args.outputs_dir, args.data_dir, args.max_epocas, args.refazer)


if __name__ == "__main__":
    main()
