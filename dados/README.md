# Dataset

The raw and processed datasets are not stored in this repository.

## Source

**2.4 GHz Indoor Channel Measurements**

UCI Machine Learning Repository
Dataset DOI: `10.24432/C5T60D`

Dataset page:

```text
https://archive.ics.uci.edu/dataset/480/2%2B4%2Bghz%2Bindoor%2Bchannel%2Bmeasurements
```

The dataset is distributed by UCI under the Creative Commons Attribution 4.0 International license.

## Download

From the project root:

```cmd
python -m ferramentas.baixar_base
```

The original ZIP file will be stored locally as:

```text
dados/canal_24ghz.zip
```

## Processing

```cmd
python -m controller.processar_base
```

The processed NumPy file will be stored locally as:

```text
dados/base_canal_24ghz.npz
```

The raw and processed files are ignored by Git.
