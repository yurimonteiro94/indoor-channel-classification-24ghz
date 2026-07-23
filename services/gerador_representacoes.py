from __future__ import annotations

import numpy as np

from model.entidades.erro_sistema import ErroSistema
from services.constantes import CODIGO_ERRO_DESCONHECIDO
from services.constantes import CODIGO_ERRO_GERACAO_ATRASOS
from services.constantes import CODIGO_ERRO_GERACAO_REPRESENTACAO
from services.constantes import CODIGO_ERRO_MATRIZ_S21_INVALIDA
from services.constantes import EPSILON_MAGNITUDE
from services.constantes import PASSO_FREQUENCIA_HZ
from services.constantes import QUANTIDADE_CANAIS_S21
from services.constantes import QUANTIDADE_PONTOS_FREQUENCIA


class GeradorRepresentacoes:
    __ultimoErro = None

    @staticmethod
    def __registrarErro(codigo:int,mensagem:str,origem:str,detalhe:str | None=None) -> None:
        GeradorRepresentacoes.__ultimoErro = ErroSistema.criar(codigo,mensagem,origem,detalhe)

    @staticmethod
    def __limparUltimoErro() -> None:
        GeradorRepresentacoes.__ultimoErro = None

    @staticmethod
    def getUltimoErro() -> ErroSistema | None:
        return GeradorRepresentacoes.__ultimoErro

    @staticmethod
    def ultimaExecucaoDeuErro() -> bool:
        if(GeradorRepresentacoes.__ultimoErro is None):
            return False

        else:
            return True

    @staticmethod
    def __validarMatrizS21(matrizS21:np.ndarray) -> bool:
        try:
            if(matrizS21.ndim != 3):
                GeradorRepresentacoes.__registrarErro(CODIGO_ERRO_MATRIZ_S21_INVALIDA,"A matriz S21 deve possuir três dimensões","GeradorRepresentacoes.__validarMatrizS21")
                return False

            elif(matrizS21.shape[0] == 0):
                GeradorRepresentacoes.__registrarErro(CODIGO_ERRO_MATRIZ_S21_INVALIDA,"A matriz S21 não pode ser vazia","GeradorRepresentacoes.__validarMatrizS21")
                return False

            elif(matrizS21.shape[1] != QUANTIDADE_PONTOS_FREQUENCIA):
                GeradorRepresentacoes.__registrarErro(CODIGO_ERRO_MATRIZ_S21_INVALIDA,"A quantidade de pontos de frequência é inválida","GeradorRepresentacoes.__validarMatrizS21",str(matrizS21.shape))
                return False

            elif(matrizS21.shape[2] != QUANTIDADE_CANAIS_S21):
                GeradorRepresentacoes.__registrarErro(CODIGO_ERRO_MATRIZ_S21_INVALIDA,"A quantidade de canais de S21 é inválida","GeradorRepresentacoes.__validarMatrizS21",str(matrizS21.shape))
                return False

            elif(not np.all(np.isfinite(matrizS21))):
                GeradorRepresentacoes.__registrarErro(CODIGO_ERRO_MATRIZ_S21_INVALIDA,"A matriz S21 contém valores inválidos","GeradorRepresentacoes.__validarMatrizS21")
                return False

            else:
                return True

        except Exception as excecao:
            GeradorRepresentacoes.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado durante a validação de S21","GeradorRepresentacoes.__validarMatrizS21",str(excecao))
            return False

    @staticmethod
    def gerarCartesiana(matrizS21:np.ndarray) -> np.ndarray | None:
        GeradorRepresentacoes.__limparUltimoErro()

        try:
            matrizS21 = np.asarray(matrizS21)

            if(not GeradorRepresentacoes.__validarMatrizS21(matrizS21)):
                return None

            else:
                return np.asarray(matrizS21,dtype=np.float32).copy()

        except Exception as excecao:
            GeradorRepresentacoes.__registrarErro(CODIGO_ERRO_GERACAO_REPRESENTACAO,"Não foi possível gerar a representação cartesiana","GeradorRepresentacoes.gerarCartesiana",str(excecao))
            return None

    @staticmethod
    def gerarPolar(matrizS21:np.ndarray) -> np.ndarray | None:
        GeradorRepresentacoes.__limparUltimoErro()

        try:
            matrizS21 = np.asarray(matrizS21)

            if(not GeradorRepresentacoes.__validarMatrizS21(matrizS21)):
                return None

            else:
                vetorComplexo = matrizS21[:,:,0] + 1j * matrizS21[:,:,1]
                magnitude = np.abs(vetorComplexo)
                magnitude = np.maximum(magnitude,EPSILON_MAGNITUDE)
                magnitudeDb = 20.0 * np.log10(magnitude)
                fase = np.angle(vetorComplexo)
                faseDesenrolada = np.unwrap(fase,axis=1)

                matrizPolar = np.empty(matrizS21.shape,dtype=np.float32)
                matrizPolar[:,:,0] = magnitudeDb.astype(np.float32)
                matrizPolar[:,:,1] = faseDesenrolada.astype(np.float32)

                return matrizPolar

        except Exception as excecao:
            GeradorRepresentacoes.__registrarErro(CODIGO_ERRO_GERACAO_REPRESENTACAO,"Não foi possível gerar a representação polar","GeradorRepresentacoes.gerarPolar",str(excecao))
            return None

    @staticmethod
    def gerarTemporal(matrizS21:np.ndarray) -> np.ndarray | None:
        GeradorRepresentacoes.__limparUltimoErro()

        try:
            matrizS21 = np.asarray(matrizS21)

            if(not GeradorRepresentacoes.__validarMatrizS21(matrizS21)):
                return None

            else:
                vetorComplexo = matrizS21[:,:,0] + 1j * matrizS21[:,:,1]
                respostaTemporal = np.fft.ifft(vetorComplexo,axis=1)

                matrizTemporal = np.empty(matrizS21.shape,dtype=np.float32)
                matrizTemporal[:,:,0] = respostaTemporal.real.astype(np.float32)
                matrizTemporal[:,:,1] = respostaTemporal.imag.astype(np.float32)

                return matrizTemporal

        except Exception as excecao:
            GeradorRepresentacoes.__registrarErro(CODIGO_ERRO_GERACAO_REPRESENTACAO,"Não foi possível gerar a representação temporal","GeradorRepresentacoes.gerarTemporal",str(excecao))
            return None

    @staticmethod
    def obterVetorAtrasos() -> np.ndarray | None:
        GeradorRepresentacoes.__limparUltimoErro()

        try:
            passoAtraso = 1.0 / (QUANTIDADE_PONTOS_FREQUENCIA * PASSO_FREQUENCIA_HZ)
            vetorAtrasos = np.arange(QUANTIDADE_PONTOS_FREQUENCIA,dtype=np.float64) * passoAtraso
            return vetorAtrasos

        except Exception as excecao:
            GeradorRepresentacoes.__registrarErro(CODIGO_ERRO_GERACAO_ATRASOS,"Não foi possível gerar o vetor de atrasos","GeradorRepresentacoes.obterVetorAtrasos",str(excecao))
            return None