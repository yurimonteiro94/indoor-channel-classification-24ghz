from __future__ import annotations

import torch
from torch import nn

from services.constantes import CNN_CANAIS_PRIMEIRA_CAMADA
from services.constantes import CNN_CANAIS_SEGUNDA_CAMADA
from services.constantes import CNN_CANAIS_TERCEIRA_CAMADA
from services.constantes import CNN_DROPOUT
from services.constantes import CNN_NEURONIOS_CAMADA_DENSA
from services.constantes import CNN_TAMANHO_POOLING_ADAPTATIVO
from services.constantes import QUANTIDADE_AMBIENTES


class ModeloCNN1D(nn.Module):
    def __init__(self):
        super().__init__()

        self.__extratorDeCaracteristicas = nn.Sequential(
            nn.Conv1d(2,CNN_CANAIS_PRIMEIRA_CAMADA,kernel_size=7,padding=3),
            nn.BatchNorm1d(CNN_CANAIS_PRIMEIRA_CAMADA),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(CNN_CANAIS_PRIMEIRA_CAMADA,CNN_CANAIS_SEGUNDA_CAMADA,kernel_size=5,padding=2),
            nn.BatchNorm1d(CNN_CANAIS_SEGUNDA_CAMADA),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(CNN_CANAIS_SEGUNDA_CAMADA,CNN_CANAIS_TERCEIRA_CAMADA,kernel_size=3,padding=1),
            nn.BatchNorm1d(CNN_CANAIS_TERCEIRA_CAMADA),
            nn.ReLU()
        )

        self.__poolingAdaptativo = nn.AdaptiveAvgPool1d(CNN_TAMANHO_POOLING_ADAPTATIVO)

        quantidadeEntradasCamadaDensa = CNN_CANAIS_TERCEIRA_CAMADA * CNN_TAMANHO_POOLING_ADAPTATIVO

        self.__classificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(quantidadeEntradasCamadaDensa,CNN_NEURONIOS_CAMADA_DENSA),
            nn.ReLU(),
            nn.Dropout(CNN_DROPOUT),
            nn.Linear(CNN_NEURONIOS_CAMADA_DENSA,QUANTIDADE_AMBIENTES)
        )

    def forward(self,tensorEntrada:torch.Tensor) -> torch.Tensor:
        tensorSaida = self.__extratorDeCaracteristicas(tensorEntrada)
        tensorSaida = self.__poolingAdaptativo(tensorSaida)
        tensorSaida = self.__classificador(tensorSaida)

        return tensorSaida

    def contarParametrosTreinaveis(self) -> int:
        quantidadeParametros = 0

        for parametro in self.parameters():
            if(parametro.requires_grad):
                quantidadeParametros = quantidadeParametros + parametro.numel()

        return quantidadeParametros