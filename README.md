# U-Net BCCD

Repositorio modular para preparar o dataset BCCD, treinar uma U-Net com
convolucoes validas e avaliar segmentacao binaria de celulas.

## Estrutura

- `download_dataset.py`: entrada principal para baixar/processar o dataset.
- `unet_bccd/data_prep.py`: download, validacao, corte em tiles e pesos.
- `unet_bccd/weights.py`: mapa de pesos da U-Net e pesos de classe opcionais.
- `unet_bccd/model.py`: arquitetura U-Net original com saida 388x388.
- `unet_bccd/dataset.py`: dataset PyTorch para tiles imagem/mascara/peso.
- `unet_bccd/train.py`: loop de treino, validacao, checkpoints e historico.
- `unet_bccd/evaluate.py`: avaliacao em teste, metricas e visualizacao.
- `tests/`: testes pequenos para pesos e shape da rede.

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

Por padrao, a saida fica em:

```text
data/BCCD_processado/
  train/original_tiles
  train/mask_tiles
  train/pesos
  val/original_tiles
  val/mask_tiles
  test/original_tiles
  test/mask_tiles
```

Para usar tambem balanceamento `wc(x)` de fundo/celula no mapa de pesos:

```powershell
python download_dataset.py --destination data --skip-download --use-class-weights
```

## Treinar

Edite os hiperparametros em:

```text
configs/train.toml
```

Depois rode:

```powershell
python -m unet_bccd.train --config configs/train.toml
```

Voce tambem pode sobrescrever qualquer valor do arquivo pelo terminal:

```powershell
python -m unet_bccd.train --config configs/train.toml --epochs 1 --batch-size 2 --lr 0.0005 --output-dir runs/debug
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
