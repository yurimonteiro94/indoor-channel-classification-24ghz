# Dados

Esta pasta contém a documentação da base utilizada pelo projeto. Os arquivos brutos, processados e preparados são mantidos apenas localmente e não são armazenados no Git.

## Fonte

**2.4 GHz Indoor Channel Measurements**

UCI Machine Learning Repository

Página da base:

[https://archive.ics.uci.edu/dataset/480/2%2B4%2Bghz%2Bindoor%2Bchannel%2Bmeasurements](https://archive.ics.uci.edu/dataset/480/2%2B4%2Bghz%2Bindoor%2Bchannel%2Bmeasurements)

DOI:

[`10.24432/C5T60D`](https://doi.org/10.24432/C5T60D)

A base é distribuída sob a licença Creative Commons Attribution 4.0 International.

## Composição da base

A base possui:

- quatro ambientes;
- 196 posições físicas em cada ambiente;
- dez medições consecutivas por posição;
- 7.840 medições no total;
- 601 pontos de frequência por medição;
- faixa de frequência entre 2,4 GHz e 2,5 GHz;
- grade espacial de \(14 \times 14\) posições em cada ambiente.

Os quatro ambientes são:

- corredor;
- laboratório;
- saguão principal;
- ginásio esportivo.

Cada arquivo CSV original contém a frequência e as partes real e imaginária dos parâmetros \(S_{11}\) e \(S_{21}\).

A implementação atual utiliza as partes real e imaginária do coeficiente de transmissão \(S_{21}\).

## Arquivos locais

### `canal_24ghz.zip`

Arquivo ZIP original obtido do UCI Machine Learning Repository.

```text
dados/canal_24ghz.zip
```

### `base_canal_24ghz.npz`

Base intermediária construída a partir dos arquivos CSV.

Ela contém:

- vetor de frequências;
- partes real e imaginária de \(S_{21}\);
- classe de cada medição;
- ambiente;
- posição física;
- repetição;
- identificador de grupo;
- caminho original do arquivo.

```text
dados/base_canal_24ghz.npz
```

### `base_preparada_24ghz.npz`

Base utilizada nos experimentos de aprendizado de máquina.

Ela contém:

- representação cartesiana normalizada;
- representação polar normalizada;
- representação temporal normalizada;
- índices de treinamento;
- índices de validação;
- índices de teste;
- parâmetros de normalização;
- metadados necessários aos experimentos.

```text
dados/base_preparada_24ghz.npz
```

## Download

A partir da raiz do projeto:

```cmd
python main.py baixar-base
```

O programa reutiliza o arquivo existente quando o ZIP já foi baixado e validado.

## Processamento

Para gerar a base intermediária:

```cmd
python main.py processar-base
```

Para criar as divisões experimentais, representações e normalizações:

```cmd
python main.py preparar-experimentos
```

## Divisão experimental

A divisão é realizada por posição física dentro de cada ambiente. As dez repetições de uma mesma posição permanecem sempre no mesmo subconjunto.

| Subconjunto | Posições por ambiente | Grupos físicos | Medições |
|---|---:|---:|---:|
| Treinamento | 137 | 548 | 5.480 |
| Validação | 29 | 116 | 1.160 |
| Teste | 30 | 120 | 1.200 |

Essa estratégia impede que repetições da mesma posição apareçam simultaneamente no treinamento e na avaliação.

A normalização é ajustada exclusivamente com os dados de treinamento. Os mesmos parâmetros são então aplicados aos conjuntos de validação e teste.

O conjunto de teste permanece reservado para a avaliação final.

## Política de versionamento

Os seguintes arquivos são ignorados pelo Git:

```text
dados/canal_24ghz.zip
dados/*.npz
dados/*.npy
```

A base original não é redistribuída neste repositório. Ela deve ser obtida diretamente do UCI Machine Learning Repository, respeitando sua licença e os requisitos de atribuição.