# Relatorio de Data Augmentation

## Por que usar Data Augmentation

Data augmentation foi usado para aumentar a variabilidade efetiva do conjunto de
treino sem exigir novas anotacoes. Isso e importante em segmentacao medica e de
microscopia porque datasets costumam ser pequenos, caros de rotular e muito
sens?veis ao equipamento, iluminacao, foco e preparacao das laminas.

## Estrategias implementadas

O projeto agora possui quatro estrategias nomeadas. Elas podem ser usadas
separadamente ou combinadas pela lista `augmentation.strategies` do arquivo TOML.

### 1. `paper`

Inclui as augmentations geometricas ja implementadas pelo grupo a partir da
U-Net original:

- deformacao elastica com grade grosseira `3x3` e sigma `10 px`;
- rotacao aleatoria em multiplos de 90 graus;
- flips horizontal e vertical.

Essas transformacoes ajudam a rede a lidar com variacoes de forma, posicao e
orientacao das celulas.

### 2. `intensity`

Inclui:

- RandomBrightnessContrast;
- CLAHE.

Essa estrategia simula variacoes reais de microscopia, como diferencas de
iluminacao e baixo contraste local entre celulas e fundo.

### 3. `acquisition_noise`

Inclui:

- GaussianBlur leve;
- GaussNoise leve.

Essa estrategia simula variacoes de foco, sensor e ruido de aquisicao.

### 4. `local_rotation`

Inclui o Random Local Rotation (RLR) definido em `unet_bccd/augmentation.py`.
Ele rotaciona uma regiao circular aleatoria da imagem. A mesma transformacao
geometrica e aplicada a imagem, mascara e mapa de pesos.

Essa estrategia pode ser util quando pequenas regioes da celula variam de
orientacao ou quando se deseja aumentar a diversidade local sem transformar o
tile inteiro.

## Garantia de alinhamento entre imagem e mascara

As transformacoes geometricas usam os mesmos parametros para:

- imagem de entrada;
- mascara correspondente;
- mapa de pesos, quando existe.

No codigo:

- `paper` usa o mesmo campo de deformacao, a mesma rotacao e os mesmos flips;
- `local_rotation` usa a mesma matriz de rotacao local e o mesmo circulo;
- imagem e pesos usam interpolacao continua;
- mascara usa nearest-neighbor para preservar classes binarias.

As transformacoes de intensidade (`intensity` e `acquisition_noise`) sao
aplicadas somente na imagem. A mascara nao deve receber brilho, contraste,
CLAHE, blur ou ruido porque ela representa labels, nao aparencia visual.

## HueSaturationValue / ColorJitter

HueSaturationValue ou ColorJitter em RGB nao foram implementados nesta etapa
porque o dataset carrega imagens em escala de cinza (`convert("L")`) e a U-Net
usa `in_channels=1`. Sem canais de cor, matiz e saturacao nao existem no tensor
de entrada.

Se o projeto passar a usar RGB (`in_channels=3`), uma quinta estrategia de
coloracao pode ser adicionada com HSV jitter leve, stain jitter ou perturbacoes
em espaco HED.

## Como executar

Sem data augmentation:

```powershell
python -m unet_bccd.train --config configs/train_no_augmentation.toml
```

Somente paper:

```powershell
python -m unet_bccd.train --config configs/train_paper_augmentation.toml
```

Todas as quatro estrategias:

```powershell
python -m unet_bccd.train --config configs/train_four_strategies.toml
```

Exemplo com somente duas estrategias:

```powershell
python -m unet_bccd.train --config configs/train_two_strategies_example.toml
```

Tambem e possivel escolher diretamente pelo terminal:

```powershell
python -m unet_bccd.train --config configs/train_paper_augmentation.toml --augmentation-strategies paper local_rotation --output-dir runs/paper_local_rotation
```

## Geracao de exemplos comparativos

Para gerar um exemplo por estrategia:

```powershell
python -m unet_bccd.preview_augmentations --data-root data/BCCD_processado --output-dir runs/augmentation_examples
```

Cada PNG contem:

- imagem original;
- mascara original;
- imagem aumentada;
- mascara aumentada.
