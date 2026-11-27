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

### Observação: arquivos duplicados com rótulos diferentes

A EDA ([notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)) encontrou três pares de arquivos idênticos byte a byte, salvos em classes diferentes:

| Arquivo | Cópia idêntica | Conjuntos |
|---|---|---|
| `glass/glass115.jpg` | `metal/metal91.jpg` | treino / validação |
| `glass/glass176.jpg` | `plastic/plastic152.jpg` | validação / treino |
| `glass/glass389.jpg` | `plastic/plastic332.jpg` | treino / treino |

Visualmente, as três fotos são garrafas de vidro. Os arquivos foram **mantidos sem alteração**, para preservar o dataset original (2.527 imagens e as contagens por classe da proposta). Eles representam menos de 0,2% dos dados e nenhum está no conjunto de teste. Ficam registrados como ruído de rótulo e limitação do dataset.
