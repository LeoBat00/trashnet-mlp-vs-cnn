"""Arquiteturas MLP e CNN.

As duas recebem a imagem uint8 (altura, largura, 3) e comecam pelas mesmas
camadas de pre-processamento:
  1. Rescaling(1/255): normalizacao para [0, 1] (sem estatisticas de dados,
     portanto sem risco de vazamento).
  2. Aumento de dados: espelhamento horizontal, rotacao de ate ~18 graus e
     zoom de ate 10%. Essas camadas do Keras so atuam com training=True,
     ou seja, apenas em model.fit sobre o treino.
"""

from tensorflow import keras
from tensorflow.keras import layers

from src.config import BN_MOMENTUM, LR, NUM_CLASSES, SEED


def _entrada_e_preprocessamento(input_shape):
    entrada = keras.Input(shape=input_shape, name="imagem")
    x = layers.Rescaling(1.0 / 255, name="normalizacao")(entrada)
    x = layers.RandomFlip("horizontal", seed=SEED, name="aug_flip")(x)
    x = layers.RandomRotation(0.05, fill_mode="nearest", seed=SEED, name="aug_rotacao")(x)
    x = layers.RandomZoom(0.1, fill_mode="nearest", seed=SEED, name="aug_zoom")(x)
    return entrada, x


def criar_mlp(input_shape):
    """Vetor de pixels -> 2 camadas ocultas (Dense+BN+ReLU+Dropout) -> softmax com 6 saidas.

    O BatchNorm nas ocultas foi necessario: sem ele a MLP nao saia do chute
    (acuracia de treino ~22%) com o aumento de dados ligado.
    """
    entrada, x = _entrada_e_preprocessamento(input_shape)
    x = layers.Flatten(name="vetor_pixels")(x)
    for n, unidades in enumerate((256, 128), start=1):
        x = layers.Dense(unidades, use_bias=False, name=f"oculta_{n}")(x)
        x = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"bn_oculta_{n}")(x)
        x = layers.Activation("relu", name=f"relu_oculta_{n}")(x)
        x = layers.Dropout(0.3, seed=SEED, name=f"dropout_{n}")(x)
    saida = layers.Dense(NUM_CLASSES, activation="softmax", name="saida")(x)
    return keras.Model(entrada, saida, name="mlp")


def _bloco_conv(x, filtros, n):
    x = layers.Conv2D(filtros, 3, padding="same", use_bias=False, name=f"conv_{n}")(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"bn_{n}")(x)
    x = layers.Activation("relu", name=f"relu_{n}")(x)
    return layers.MaxPooling2D(2, name=f"pool_{n}")(x)


def criar_cnn(input_shape, filtros=(32, 64, 128)):
    """3 blocos conv+BN+ReLU+pool -> Global Average Pooling -> softmax.

    O Global Average Pooling faz o numero de parametros ser o mesmo em
    qualquer resolucao: so a entrada muda entre os experimentos 3, 5 e 6.
    """
    entrada, x = _entrada_e_preprocessamento(input_shape)
    for n, f in enumerate(filtros, start=1):
        x = _bloco_conv(x, f, n)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, seed=SEED)(x)
    saida = layers.Dense(NUM_CLASSES, activation="softmax", name="saida")(x)
    return keras.Model(entrada, saida, name="cnn")


def criar_modelo(tipo, input_shape):
    """Cria e compila o modelo: Adam + entropia cruzada (rotulos inteiros)."""
    modelo = {"mlp": criar_mlp, "cnn": criar_cnn}[tipo](input_shape)
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR[tipo]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo
