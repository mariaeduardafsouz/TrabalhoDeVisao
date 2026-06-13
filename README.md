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

Se o modelo colapsar para prever somente fundo, use a versao com peso de classe
explicito na loss:

```powershell
python -m unet_bccd.train --config configs/train_four_strategies_class_weighted.toml
```

Tambem e possivel passar os pesos diretamente pela linha de comando. A ordem e
`[fundo, celula]`:

```powershell
python -m unet_bccd.train --config configs/train_four_strategies.toml --class-weights 1.0 4.0 --output-dir runs/unet_four_strategies_class_weighted
```

Com somente duas estrategias, edite `configs/train_two_strategies_example.toml` ou rode:

```powershell
python -m unet_bccd.train --config configs/train_paper_augmentation.toml --augmentation-strategies paper local_rotation --output-dir runs/paper_local_rotation
```

Para continuar um treino a partir de um checkpoint salvo, use `--resume-checkpoint`.
Por exemplo, para continuar depois do checkpoint da epoca 15:

```powershell
python -m unet_bccd.train --config configs/train_four_strategies.toml --resume-checkpoint runs/unet_four_strategies_v2/checkpoints/unet_epoch_15.pth --output-dir runs/unet_four_strategies_v2
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

O treinamento suporta **early stopping**: se a `val_loss` nao melhorar por
`early_stopping_patience` epocas consecutivas, o treino para automaticamente.
O melhor modelo fica salvo em `runs/unet/unet_best.pth`. Para desativar,
use `--early-stopping-patience 0`.

```powershell
python -m unet_bccd.train --config configs/train.toml --early-stopping-patience 10
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
