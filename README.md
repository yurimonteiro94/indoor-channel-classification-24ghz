# Indoor Channel Classification at 2.4 GHz

Classification of indoor environments from real wireless-channel measurements using neural networks and nonlinear optimization.

The project is being developed for PRO6006 — Nonlinear Optimization Methods with Applications in Machine Learning.

## Research problem

The objective is to determine which wireless-channel representation and neural-network configuration provide the best indoor-environment classification performance with the lowest computational complexity.

The four environments are:

- corridor
- laboratory
- main lobby
- sports hall

Each measurement contains 601 complex samples of the transmission coefficient \(S_{21}(f)\), measured between 2.4 GHz and 2.5 GHz.

## Dataset

The dataset contains:

- 4 environments
- 196 physical positions per environment
- 10 consecutive measurements per position
- 7,840 measurements
- 601 frequency samples per measurement

The positions form a 14 by 14 spatial grid in each environment.

Measurements originating from the same physical position must not be divided between training, validation and test subsets. The project therefore preserves a group identifier for each physical position.

Dataset source:

```text
https://archive.ics.uci.edu/dataset/480/2%2B4%2Bghz%2Bindoor%2Bchannel%2Bmeasurements
```

Dataset DOI:

```text
10.24432/C5T60D
```

## Current implementation

The current pipeline:

1. downloads or reuses the original UCI ZIP file
2. reads the CSV files directly from the ZIP
3. identifies environment, physical position and repetition
4. extracts the real and imaginary components of \(S_{21}\)
5. validates the frequency grid and measurement dimensions
6. builds a compressed NumPy dataset
7. preserves physical-position groups for leakage-free experiments

## Environment

```cmd
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

## Download the dataset

```cmd
python -m ferramentas.baixar_base
```

## Run the tests

```cmd
python -m unittest discover -s test -p "test_*.py" -v
```

## Process the complete dataset

```cmd
python -m controller.processar_base
```

The processed dataset is generated locally at:

```text
dados/base_canal_24ghz.npz
```

Raw and processed datasets are intentionally excluded from Git.