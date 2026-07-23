from __future__ import annotations

import numpy as np

from model.entidades.erro_sistema import ErroSistema
from services.constantes import CHAVE_DADOS_NORMALIZADOS
from services.constantes import CHAVE_DESVIOS
from services.constantes import CHAVE_MEDIAS
from services.constantes import CODIGO_ERRO_DESCONHECIDO
from services.constantes import CODIGO_ERRO_INDICES_NORMALIZACAO_INVALIDOS
from services.constantes import CODIGO_ERRO_MATRIZ_NORMALIZACAO_INVALIDA
from services.constantes import CODIGO_ERRO_NORMALIZACAO
from services.constantes import CODIGO_ERRO_PARAMETROS_NORMALIZACAO_INVALIDOS
from services.constantes import DESVIO_PADRAO_MINIMO


class NormalizadorDados:
    __ultimoErro = None

    @staticmethod
    def __registrarErro(codigo:int,mensagem:str,origem:str,detalhe:str | None=None) -> None:
        NormalizadorDados.__ultimoErro = ErroSistema.criar(codigo,mensagem,origem,detalhe)

    @staticmethod
    def __limparUltimoErro() -> None:
        NormalizadorDados.__ultimoErro = None

    @staticmethod
    def getUltimoErro() -> ErroSistema | None:
        return NormalizadorDados.__ultimoErro

    @staticmethod
    def ultimaExecucaoDeuErro() -> bool:
        if(NormalizadorDados.__ultimoErro is None):
            return False

        else:
            return True

    @staticmethod
    def __validarMatriz(matrizDados:np.ndarray) -> bool:
        try:
            if(matrizDados.ndim != 3):
                NormalizadorDados.__registrarErro(CODIGO_ERRO_MATRIZ_NORMALIZACAO_INVALIDA,"A matriz de dados deve possuir três dimensões","NormalizadorDados.__validarMatriz")
                return False

            elif(matrizDados.shape[0] == 0):
                NormalizadorDados.__registrarErro(CODIGO_ERRO_MATRIZ_NORMALIZACAO_INVALIDA,"A matriz de dados não pode ser vazia","NormalizadorDados.__validarMatriz")
                return False

            elif(matrizDados.shape[1] == 0 or matrizDados.shape[2] == 0):
                NormalizadorDados.__registrarErro(CODIGO_ERRO_MATRIZ_NORMALIZACAO_INVALIDA,"A matriz de dados possui uma dimensão vazia","NormalizadorDados.__validarMatriz")
                return False

            elif(not np.all(np.isfinite(matrizDados))):
                NormalizadorDados.__registrarErro(CODIGO_ERRO_MATRIZ_NORMALIZACAO_INVALIDA,"A matriz de dados contém valores inválidos","NormalizadorDados.__validarMatriz")
                return False

            else:
                return True

        except Exception as excecao:
            NormalizadorDados.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado durante a validação da matriz","NormalizadorDados.__validarMatriz",str(excecao))
            return False

    @staticmethod
    def __validarIndices(vetorIndices:np.ndarray,quantidadeDeMedicoes:int) -> bool:
        try:
            if(vetorIndices.ndim != 1):
                NormalizadorDados.__registrarErro(CODIGO_ERRO_INDICES_NORMALIZACAO_INVALIDOS,"O vetor de índices deve possuir uma dimensão","NormalizadorDados.__validarIndices")
                return False

            elif(vetorIndices.size == 0):
                NormalizadorDados.__registrarErro(CODIGO_ERRO_INDICES_NORMALIZACAO_INVALIDOS,"O vetor de índices não pode ser vazio","NormalizadorDados.__validarIndices")
                return False

            elif(not np.issubdtype(vetorIndices.dtype,np.integer)):
                NormalizadorDados.__registrarErro(CODIGO_ERRO_INDICES_NORMALIZACAO_INVALIDOS,"O vetor de índices deve possuir valores inteiros","NormalizadorDados.__validarIndices")
                return False

            elif(np.any(vetorIndices < 0) or np.any(vetorIndices >= quantidadeDeMedicoes)):
                NormalizadorDados.__registrarErro(CODIGO_ERRO_INDICES_NORMALIZACAO_INVALIDOS,"O vetor de índices contém valores inválidos","NormalizadorDados.__validarIndices")
                return False

            elif(np.unique(vetorIndices).size != vetorIndices.size):
                NormalizadorDados.__registrarErro(CODIGO_ERRO_INDICES_NORMALIZACAO_INVALIDOS,"O vetor de índices contém valores repetidos","NormalizadorDados.__validarIndices")
                return False

            else:
                return True

        except Exception as excecao:
            NormalizadorDados.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado durante a validação dos índices","NormalizadorDados.__validarIndices",str(excecao))
            return False

    @staticmethod
    def calcularParametros(matrizDados:np.ndarray,indicesTreino:np.ndarray) -> dict[str,np.ndarray] | None:
        NormalizadorDados.__limparUltimoErro()

        try:
            matrizDados = np.asarray(matrizDados)
            indicesTreino = np.asarray(indicesTreino)

            if(not NormalizadorDados.__validarMatriz(matrizDados)):
                return None

            elif(not NormalizadorDados.__validarIndices(indicesTreino,matrizDados.shape[0])):
                return None

            else:
                quantidadeDeCanais = matrizDados.shape[2]
                vetorMedias = np.empty(quantidadeDeCanais,dtype=np.float64)
                vetorDesvios = np.empty(quantidadeDeCanais,dtype=np.float64)

                for indiceCanal in range(quantidadeDeCanais):
                    dadosDoCanal = matrizDados[indicesTreino,:,indiceCanal]
                    media = float(np.mean(dadosDoCanal,dtype=np.float64))
                    desvio = float(np.std(dadosDoCanal,dtype=np.float64))

                    if(not np.isfinite(media)):
                        NormalizadorDados.__registrarErro(CODIGO_ERRO_PARAMETROS_NORMALIZACAO_INVALIDOS,"A média calculada é inválida","NormalizadorDados.calcularParametros",str(indiceCanal))
                        return None

                    elif(not np.isfinite(desvio)):
                        NormalizadorDados.__registrarErro(CODIGO_ERRO_PARAMETROS_NORMALIZACAO_INVALIDOS,"O desvio padrão calculado é inválido","NormalizadorDados.calcularParametros",str(indiceCanal))
                        return None

                    else:
                        if(desvio < DESVIO_PADRAO_MINIMO):
                            desvio = 1.0

                        vetorMedias[indiceCanal] = media
                        vetorDesvios[indiceCanal] = desvio

                return {
                    CHAVE_MEDIAS: vetorMedias,
                    CHAVE_DESVIOS: vetorDesvios
                }

        except Exception as excecao:
            NormalizadorDados.__registrarErro(CODIGO_ERRO_NORMALIZACAO,"Não foi possível calcular os parâmetros de normalização","NormalizadorDados.calcularParametros",str(excecao))
            return None

    @staticmethod
    def aplicarNormalizacao(matrizDados:np.ndarray,vetorMedias:np.ndarray,vetorDesvios:np.ndarray) -> np.ndarray | None:
        NormalizadorDados.__limparUltimoErro()

        try:
            matrizDados = np.asarray(matrizDados)
            vetorMedias = np.asarray(vetorMedias)
            vetorDesvios = np.asarray(vetorDesvios)

            if(not NormalizadorDados.__validarMatriz(matrizDados)):
                return None

            elif(vetorMedias.ndim != 1 or vetorDesvios.ndim != 1):
                NormalizadorDados.__registrarErro(CODIGO_ERRO_PARAMETROS_NORMALIZACAO_INVALIDOS,"Os parâmetros de normalização devem possuir uma dimensão","NormalizadorDados.aplicarNormalizacao")
                return None

            elif(vetorMedias.size != matrizDados.shape[2] or vetorDesvios.size != matrizDados.shape[2]):
                NormalizadorDados.__registrarErro(CODIGO_ERRO_PARAMETROS_NORMALIZACAO_INVALIDOS,"A quantidade de parâmetros não corresponde à quantidade de canais","NormalizadorDados.aplicarNormalizacao")
                return None

            elif(not np.all(np.isfinite(vetorMedias)) or not np.all(np.isfinite(vetorDesvios))):
                NormalizadorDados.__registrarErro(CODIGO_ERRO_PARAMETROS_NORMALIZACAO_INVALIDOS,"Os parâmetros de normalização contêm valores inválidos","NormalizadorDados.aplicarNormalizacao")
                return None

            elif(np.any(vetorDesvios <= 0.0)):
                NormalizadorDados.__registrarErro(CODIGO_ERRO_PARAMETROS_NORMALIZACAO_INVALIDOS,"Os desvios padrão devem ser positivos","NormalizadorDados.aplicarNormalizacao")
                return None

            else:
                mediasRedimensionadas = vetorMedias.reshape(1,1,vetorMedias.size)
                desviosRedimensionados = vetorDesvios.reshape(1,1,vetorDesvios.size)
                matrizNormalizada = (matrizDados - mediasRedimensionadas) / desviosRedimensionados
                return np.asarray(matrizNormalizada,dtype=np.float32)

        except Exception as excecao:
            NormalizadorDados.__registrarErro(CODIGO_ERRO_NORMALIZACAO,"Não foi possível aplicar a normalização","NormalizadorDados.aplicarNormalizacao",str(excecao))
            return None

    @staticmethod
    def normalizarPorCanais(matrizDados:np.ndarray,indicesTreino:np.ndarray) -> dict[str,np.ndarray] | None:
        NormalizadorDados.__limparUltimoErro()

        try:
            dicionarioParametros = NormalizadorDados.calcularParametros(matrizDados,indicesTreino)

            if(dicionarioParametros is None):
                return None

            else:
                vetorMedias = dicionarioParametros[CHAVE_MEDIAS]
                vetorDesvios = dicionarioParametros[CHAVE_DESVIOS]
                matrizNormalizada = NormalizadorDados.aplicarNormalizacao(matrizDados,vetorMedias,vetorDesvios)

                if(matrizNormalizada is None):
                    return None

                else:
                    return {
                        CHAVE_DADOS_NORMALIZADOS: matrizNormalizada,
                        CHAVE_MEDIAS: vetorMedias,
                        CHAVE_DESVIOS: vetorDesvios
                    }

        except Exception as excecao:
            NormalizadorDados.__registrarErro(CODIGO_ERRO_NORMALIZACAO,"Não foi possível normalizar os dados por canais","NormalizadorDados.normalizarPorCanais",str(excecao))
            return None