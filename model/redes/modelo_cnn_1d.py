from __future__ import annotations

import torch
from torch import nn

from services.constantes import CNN_CANAIS_PRIMEIRA_CAMADA
from services.constantes import CNN_CANAIS_SEGUNDA_CAMADA
from services.constantes import CNN_CANAIS_TERCEIRA_CAMADA
from services.constantes import CNN_DECAIMENTO_PESOS
from services.constantes import CNN_DROPOUT
from services.constantes import CNN_NEURONIOS_CAMADA_DENSA
from services.constantes import CNN_TAMANHO_LOTE
from services.constantes import CNN_TAMANHO_POOLING_ADAPTATIVO
from services.constantes import CNN_TAXA_APRENDIZADO
from services.constantes import QUANTIDADE_AMBIENTES


class ConfiguracaoCNN:
    def __init__(self,canaisPrimeiraCamada:int,canaisSegundaCamada:int,canaisTerceiraCamada:int,tamanhoPoolingAdaptativo:int,neuroniosCamadaDensa:int,dropout:float,tamanhoLote:int,taxaAprendizado:float,decaimentoPesos:float):
        self.__canaisPrimeiraCamada = canaisPrimeiraCamada
        self.__canaisSegundaCamada = canaisSegundaCamada
        self.__canaisTerceiraCamada = canaisTerceiraCamada
        self.__tamanhoPoolingAdaptativo = tamanhoPoolingAdaptativo
        self.__neuroniosCamadaDensa = neuroniosCamadaDensa
        self.__dropout = dropout
        self.__tamanhoLote = tamanhoLote
        self.__taxaAprendizado = taxaAprendizado
        self.__decaimentoPesos = decaimentoPesos

    @staticmethod
    def criar(canaisPrimeiraCamada:int,canaisSegundaCamada:int,canaisTerceiraCamada:int,tamanhoPoolingAdaptativo:int,neuroniosCamadaDensa:int,dropout:float,tamanhoLote:int,taxaAprendizado:float,decaimentoPesos:float) -> ConfiguracaoCNN | None:
        try:
            if(not isinstance(canaisPrimeiraCamada,int) or isinstance(canaisPrimeiraCamada,bool)):
                return None

            elif(not isinstance(canaisSegundaCamada,int) or isinstance(canaisSegundaCamada,bool)):
                return None

            elif(not isinstance(canaisTerceiraCamada,int) or isinstance(canaisTerceiraCamada,bool)):
                return None

            elif(not isinstance(tamanhoPoolingAdaptativo,int) or isinstance(tamanhoPoolingAdaptativo,bool)):
                return None

            elif(not isinstance(neuroniosCamadaDensa,int) or isinstance(neuroniosCamadaDensa,bool)):
                return None

            elif(not isinstance(tamanhoLote,int) or isinstance(tamanhoLote,bool)):
                return None

            elif(canaisPrimeiraCamada < 1 or canaisSegundaCamada < 1 or canaisTerceiraCamada < 1):
                return None

            elif(tamanhoPoolingAdaptativo < 1 or neuroniosCamadaDensa < 1 or tamanhoLote < 1):
                return None

            elif(not isinstance(dropout,(int,float)) or isinstance(dropout,bool)):
                return None

            elif(not isinstance(taxaAprendizado,(int,float)) or isinstance(taxaAprendizado,bool)):
                return None

            elif(not isinstance(decaimentoPesos,(int,float)) or isinstance(decaimentoPesos,bool)):
                return None

            elif(dropout < 0.0 or dropout >= 1.0):
                return None

            elif(taxaAprendizado <= 0.0):
                return None

            elif(decaimentoPesos < 0.0):
                return None

            else:
                return ConfiguracaoCNN(canaisPrimeiraCamada,canaisSegundaCamada,canaisTerceiraCamada,tamanhoPoolingAdaptativo,neuroniosCamadaDensa,float(dropout),tamanhoLote,float(taxaAprendizado),float(decaimentoPesos))

        except Exception:
            return None

    @staticmethod
    def criarPadrao() -> ConfiguracaoCNN:
        return ConfiguracaoCNN(CNN_CANAIS_PRIMEIRA_CAMADA,CNN_CANAIS_SEGUNDA_CAMADA,CNN_CANAIS_TERCEIRA_CAMADA,CNN_TAMANHO_POOLING_ADAPTATIVO,CNN_NEURONIOS_CAMADA_DENSA,CNN_DROPOUT,CNN_TAMANHO_LOTE,CNN_TAXA_APRENDIZADO,CNN_DECAIMENTO_PESOS)

    def getCanaisPrimeiraCamada(self) -> int:
        return self.__canaisPrimeiraCamada

    def getCanaisSegundaCamada(self) -> int:
        return self.__canaisSegundaCamada

    def getCanaisTerceiraCamada(self) -> int:
        return self.__canaisTerceiraCamada

    def getTamanhoPoolingAdaptativo(self) -> int:
        return self.__tamanhoPoolingAdaptativo

    def getNeuroniosCamadaDensa(self) -> int:
        return self.__neuroniosCamadaDensa

    def getDropout(self) -> float:
        return self.__dropout

    def getTamanhoLote(self) -> int:
        return self.__tamanhoLote

    def getTaxaAprendizado(self) -> float:
        return self.__taxaAprendizado

    def getDecaimentoPesos(self) -> float:
        return self.__decaimentoPesos

    def paraDicionario(self) -> dict[str,int | float]:
        return {
            "canais_primeira_camada": self.__canaisPrimeiraCamada,
            "canais_segunda_camada": self.__canaisSegundaCamada,
            "canais_terceira_camada": self.__canaisTerceiraCamada,
            "tamanho_pooling_adaptativo": self.__tamanhoPoolingAdaptativo,
            "neuronios_camada_densa": self.__neuroniosCamadaDensa,
            "dropout": self.__dropout,
            "tamanho_lote": self.__tamanhoLote,
            "taxa_aprendizado": self.__taxaAprendizado,
            "decaimento_pesos": self.__decaimentoPesos
        }


class ModeloCNN1D(nn.Module):
    def __init__(self,configuracao:ConfiguracaoCNN | None=None):
        super().__init__()

        if(configuracao is None):
            configuracao = ConfiguracaoCNN.criarPadrao()

        self.__configuracao = configuracao

        self.__extratorDeCaracteristicas = nn.Sequential(
            nn.Conv1d(2,configuracao.getCanaisPrimeiraCamada(),kernel_size=7,padding=3),
            nn.BatchNorm1d(configuracao.getCanaisPrimeiraCamada()),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(configuracao.getCanaisPrimeiraCamada(),configuracao.getCanaisSegundaCamada(),kernel_size=5,padding=2),
            nn.BatchNorm1d(configuracao.getCanaisSegundaCamada()),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(configuracao.getCanaisSegundaCamada(),configuracao.getCanaisTerceiraCamada(),kernel_size=3,padding=1),
            nn.BatchNorm1d(configuracao.getCanaisTerceiraCamada()),
            nn.ReLU()
        )

        self.__poolingAdaptativo = nn.AdaptiveAvgPool1d(configuracao.getTamanhoPoolingAdaptativo())

        quantidadeEntradasCamadaDensa = configuracao.getCanaisTerceiraCamada() * configuracao.getTamanhoPoolingAdaptativo()

        self.__classificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(quantidadeEntradasCamadaDensa,configuracao.getNeuroniosCamadaDensa()),
            nn.ReLU(),
            nn.Dropout(configuracao.getDropout()),
            nn.Linear(configuracao.getNeuroniosCamadaDensa(),QUANTIDADE_AMBIENTES)
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

    def getConfiguracao(self) -> ConfiguracaoCNN:
        return self.__configuracao