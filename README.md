# TrashNet: MLP vs CNN

Classificação de imagens de resíduos em 6 classes (glass, paper, cardboard, plastic, metal, trash) com redes neurais, usando o dataset TrashNet. Atividade final da disciplina de Redes Neurais Artificiais.

## Objetivo

Comparar três aspectos do problema:

1. **MLP vs CNN** em imagens de 128×96.
2. **Desequilíbrio de classes**: o efeito de pesos de classe, principalmente na classe `trash` (só 137 imagens).
3. **Resolução na CNN**: se 256×192 e 512×384 trazem ganho que compense o custo de treinamento.

## Experimentos e resultados

Divisão estratificada 70/15/15 com semente 42 (`results/split.csv`), a mesma nos seis experimentos. Treino ajusta, **F1 macro de validação escolhe**, teste usado só no fim. Hardware: Google Colab, GPU Tesla T4, lote 32.

| # | Modelo | Resolução | Pesos de classe | F1 macro (val) | Acurácia (teste) | F1 macro (teste) | Parâmetros | Tempo total |
|---|--------|-----------|-----------------|---|---|---|---|---|
| 1 | MLP | 128×96 | Não | 0,551 | 0,537 | 0,506 | 9.472.262 | 45 s |
| 2 | MLP | 128×96 | Sim | **0,560** | 0,550 | 0,535 | 9.472.262 | 58 s |
| 3 | CNN | 128×96 | Não | 0,528 | 0,553 | 0,509 | 94.694 | 39 s |
| 4 | CNN | 128×96 | Sim | 0,523 | 0,484 | 0,463 | 94.694 | 41 s |
| 5 | CNN | 256×192 | Não | 0,541 | 0,529 | 0,511 | 94.694 | 2,2 min |
| 6 | CNN | 512×384 | Não | 0,541 | 0,597 | 0,537 | 94.694 | 13,6 min |

Modelo selecionado (maior F1 macro de validação): **experimento 2**. Tabelas completas (custo, métricas por classe, efeito dos pesos) em [`results/tables/`](results/tables/) e figuras em [`results/figures/`](results/figures/).

## Estrutura do repositório

```
trashnet-mlp-vs-cnn/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py      # semente, classes, divisão, hiperparâmetros e os 6 experimentos
│   ├── dataset.py     # listagem das imagens
│   ├── split.py       # divisão estratificada 70/15/15 (gera e verifica results/split.csv)
│   ├── data.py        # leitura RGB, redimensionamento e pesos de classe
│   ├── models.py      # MLP e CNN (normalização /255 e aumento de dados dentro do modelo)
│   ├── train.py       # treino, early stopping, medição de custo, retomada
│   ├── evaluate.py    # métricas, matriz de confusão, avaliação dos modelos salvos
│   ├── report.py      # leitura e exportação dos resultados
│   └── viz.py         # estilo das figuras
├── notebooks/
│   ├── 01_eda.ipynb          # análise exploratória (roda local, sem GPU)
│   ├── 02_colab_run.ipynb    # treino dos 6 experimentos no Colab (GPU T4)
│   └── 03_resultados.ipynb   # tabelas e figuras a partir de results/ (sem treino)
└── results/
    ├── split.csv              # divisão usada em todos os experimentos
    ├── experimentos/<exp>/    # config.json, cost.json, history.json, metrics.json, predictions.csv
    ├── tables/                # tabelas do artigo (CSV)
    └── figures/               # figuras da EDA e dos resultados
```

## Dados

- Repositório original: [garythung/trashnet](https://github.com/garythung/trashnet)
- Arquivo usado: [dataset-resized.zip](https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip?download=true) (2.527 imagens, 512×384, RGB)
- Artigo de referência: [Yang e Thung (2016)](https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf)

O dataset não está versionado. Para rodar localmente, descompacte o zip em `data/` (fica `data/dataset-resized/<classe>/...`).

### Observação: arquivos duplicados com rótulos diferentes

A EDA ([notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)) encontrou três pares de arquivos idênticos byte a byte, salvos em classes diferentes:

| Arquivo | Cópia idêntica | Conjuntos |
|---|---|---|
| `glass/glass115.jpg` | `metal/metal91.jpg` | treino / validação |
| `glass/glass176.jpg` | `plastic/plastic152.jpg` | validação / treino |
| `glass/glass389.jpg` | `plastic/plastic332.jpg` | treino / treino |

Visualmente, as três fotos são garrafas de vidro. Os arquivos foram **mantidos sem alteração**, para preservar o dataset original (2.527 imagens e as contagens por classe da proposta). Eles representam menos de 0,2% dos dados e nenhum está no conjunto de teste. Ficam registrados como ruído de rótulo e limitação do dataset.

## Modelos treinados

Os seis modelos (`model.keras`) estão em uma pasta pública do Google Drive, uma subpasta por experimento:

**https://drive.google.com/drive/folders/1Pgs0jxoEaaMP1pZEdzd7pugxnWIOzz24?usp=sharing**

Para avaliar sem treinar, baixe as subpastas para `outputs/` (fica `outputs/<experimento>/model.keras`).

## Instalação

Python 3.12 ou 3.13 (o TensorFlow 2.21 ainda não suporta 3.14).

```bash
git clone https://github.com/LeoBat00/trashnet-mlp-vs-cnn.git
cd trashnet-mlp-vs-cnn
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Execução

### 1. Divisão dos dados (já incluída em `results/split.csv`)

```bash
python -m src.split --verificar   # confere: sem vazamento, estratificado
python -m src.split               # gera de novo (sai idêntica, semente 42)
```

### 2. Análise exploratória

Abrir e executar [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) (CPU basta).

### 3. Treino

No Colab com GPU T4: abrir [`notebooks/02_colab_run.ipynb`](notebooks/02_colab_run.ipynb) pelo GitHub e rodar as células. Localmente (com GPU ou paciência):

```bash
python -m src.train --exp all                  # os 6 experimentos
python -m src.train --exp exp3_cnn_128x96      # um experimento
```

Cada experimento salva em `outputs/<experimento>/` e é pulado se já tiver `metrics.json` (retomada). Caminhos podem ser trocados com `TRASHNET_DATA_DIR` e `TRASHNET_OUTPUTS_DIR`.

### 4. Avaliação sem novo treino

Com os modelos do Drive em `outputs/`:

```bash
python -m src.evaluate                         # recalcula as métricas de teste dos 6 modelos
python -m src.evaluate --exp exp2_mlp_128x96_pesos
```

O comando carrega cada `model.keras`, prevê o conjunto de teste e compara com `results/.../metrics.json`.

### 5. Tabelas e figuras

Executar [`notebooks/03_resultados.ipynb`](notebooks/03_resultados.ipynb). Ele lê só `results/experimentos/` (não precisa dos modelos nem de GPU; para a figura de erros precisa das imagens em `data/`).

## Decisões principais

- **Pré-processamento:** RGB, redimensionamento bilinear a partir de 512×384, normalização /255 dentro do modelo.
- **Aumento de dados (só no treino):** espelhamento horizontal, rotação até ~18° e zoom até 10%, como camadas Keras ativas apenas em `fit`.
- **MLP:** pixels achatados → 2 camadas ocultas (256 e 128 unidades, BatchNorm, ReLU, Dropout 0,3) → softmax 6. Adam, LR 1e-4.
- **CNN:** 3 blocos Conv 3×3 (32/64/128 filtros) + BatchNorm + ReLU + MaxPool → Global Average Pooling → Dropout 0,3 → softmax 6. Adam, LR 1e-3. Mesmo número de parâmetros nas três resoluções.
- **Treino:** entropia cruzada, lote 32, até 60 épocas, early stopping em `val_loss` (paciência 10, `restore_best_weights`), semente 42 e operações determinísticas.
- **Pesos de classe (exp. 2 e 4):** `n_treino / (6 · n_treino,c)`, só com o treino (`trash` ≈ 3,07).
- **Métricas:** `zero_division=0` — classe nunca prevista conta precisão 0, o que penaliza o modelo em vez de esconder o problema.

### Primeira rodada e correção

Uma primeira rodada (guardada no Drive como `outputs_v1`) mostrou, **nas curvas de treino e validação**, dois problemas de implementação: a MLP sem BatchNorm não aprendia nem o treino (acurácia de treino ≈ 22%, prevendo só `paper`) e o momento padrão do BatchNorm (0,99) deixava a validação instável. Foram adicionados BatchNorm às camadas ocultas da MLP e momento 0,9 nos dois modelos (commit `bdb78ed`), e os seis experimentos foram treinados de novo. Os resultados deste README são da segunda rodada.
