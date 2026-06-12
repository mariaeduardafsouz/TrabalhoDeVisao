# U-Net BCCD

Repositorio modular para preparar o dataset BCCD, treinar uma U-Net com
convolucoes validas e avaliar segmentacao binaria de celulas.

## Estrutura

- `download_dataset.py`: entrada principal para baixar/processar o dataset.
- `unet_bccd/data_prep.py`: download, validacao, corte em tiles e pesos.
- `unet_bccd/weights.py`: mapa de pesos da U-Net e pesos de classe opcionais.
- `unet_bccd/transforms.py`: estrategias de data augmentation.
- `unet_bccd/augmentation.py`: Random Local Rotation (RLR).
- `unet_bccd/preview_augmentations.py`: gera exemplos comparativos.
- `unet_bccd/model.py`: arquitetura U-Net original com saida 388x388.
- `unet_bccd/dataset.py`: dataset PyTorch para tiles imagem/mascara/peso.
- `unet_bccd/train.py`: loop de treino, validacao, checkpoints e historico.
- `unet_bccd/evaluate.py`: avaliacao em teste, metricas e visualizacao.
- `tests/`: testes pequenos para pesos, modelo e transforms.

## Instalacao

```powershell
python -m pip install -r requirements.txt
```

## Preparar dados

Para baixar do Kaggle e processar:

```powershell
python download_dataset.py --destination data
```

Se o dataset ja estiver baixado em `data/BCCD Dataset with mask`, pule o download:

```powershell
python download_dataset.py --destination data --skip-download
```

Para reconstruir a pasta processada do zero:

```powershell
python download_dataset.py --destination data --skip-download --overwrite
```

## Treinar

Sem data augmentation:

```powershell
python -m unet_bccd.train --config configs/train_no_augmentation.toml
```

Com as augmentations do paper da U-Net:

```powershell
python -m unet_bccd.train --config configs/train_paper_augmentation.toml
```

Com as quatro estrategias de augmentation:

```powershell
python -m unet_bccd.train --config configs/train_four_strategies.toml
```

Com somente duas estrategias, edite `configs/train_two_strategies_example.toml` ou rode:

```powershell
python -m unet_bccd.train --config configs/train_paper_augmentation.toml --augmentation-strategies paper local_rotation --output-dir runs/paper_local_rotation
```

Estrategias disponiveis:

- `paper`
- `intensity`
- `acquisition_noise`
- `local_rotation`

## Exemplos visuais de augmentation

```powershell
python -m unet_bccd.preview_augmentations --data-root data/BCCD_processado --output-dir runs/augmentation_examples
```

Cada imagem gerada compara:

- imagem original;
- mascara original;
- imagem aumentada;
- mascara aumentada.

O relatorio esta em:

```text
docs/data_augmentation_report.md
```

## Avaliar

```powershell
python -m unet_bccd.evaluate --data-root data/BCCD_processado --model-path runs/unet/unet_final.pth
```

Os resultados sao salvos em `runs/unet/eval/metrics.json` e, quando possivel,
`runs/unet/eval/predictions.png`.

## Testes

```powershell
python -m pytest -q
```

O teste de modelo instancia a U-Net completa, entao pode ser mais lento em CPU.
