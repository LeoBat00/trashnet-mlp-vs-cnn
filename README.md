# TrashNet: MLP vs CNN

Classificação de imagens de resíduos em 6 classes (glass, paper, cardboard, plastic, metal, trash) com redes neurais, usando o dataset TrashNet. Atividade final da disciplina de Redes Neurais Artificiais.

## Objetivo

Comparar três aspectos do problema:

1. **MLP vs CNN** em imagens de 128×96.
2. **Desequilíbrio de classes**: o efeito de pesos de classe, principalmente na classe `trash` (só 137 imagens).
3. **Resolução na CNN**: se 256×192 e 512×384 trazem ganho que compense o custo de treinamento.

## Experimentos

| # | Modelo | Resolução | Pesos de classe |
|---|--------|-----------|-----------------|
| 1 | MLP | 128×96 | Não |
| 2 | MLP | 128×96 | Sim |
| 3 | CNN | 128×96 | Não |
| 4 | CNN | 128×96 | Sim |
| 5 | CNN | 256×192 | Não |
| 6 | CNN | 512×384 | Não |

## Dataset

- Repositório original: [garythung/trashnet](https://github.com/garythung/trashnet)
- Arquivo usado: [dataset-resized.zip](https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip?download=true) (2.527 imagens, 512×384)
- Artigo de referência: [Yang e Thung (2016)](https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf)

O dataset não está versionado neste repositório.
