from __future__ import annotations

import numpy as np

from model.entidades.erro_sistema import ErroSistema
from services.constantes import CODIGO_ERRO_CAMINHO_MEDICAO_INVALIDO
from services.constantes import CODIGO_ERRO_CLASSE_INVALIDA
from services.constantes import CODIGO_ERRO_CRIACAO_MEDICAO
from services.constantes import CODIGO_ERRO_DESCONHECIDO
from services.constantes import CODIGO_ERRO_DIMENSAO_INVALIDA
from services.constantes import CODIGO_ERRO_GRADE_FREQUENCIAS_INVALIDA
from services.constantes import CODIGO_ERRO_GRUPO_INVALIDO
from services.constantes import CODIGO_ERRO_NOME_AMBIENTE_INVALIDO
from services.constantes import CODIGO_ERRO_POSICAO_INVALIDA
from services.constantes import CODIGO_ERRO_QUANTIDADE_PONTOS_INVALIDA
from services.constantes import CODIGO_ERRO_REPETICAO_INVALIDA
from services.constantes import CODIGO_ERRO_VALOR_NUMERICO_INVALIDO
from services.constantes import FREQUENCIA_FINAL_HZ
from services.constantes import FREQUENCIA_INICIAL_HZ
from services.constantes import QUANTIDADE_AMBIENTES
from services.constantes import QUANTIDADE_GRUPOS_TOTAL
from services.constantes import QUANTIDADE_PONTOS_FREQUENCIA
from services.constantes import QUANTIDADE_REPETICOES_POR_POSICAO
from services.constantes import TOLERANCIA_FREQUENCIA_HZ


class MedicaoCanal:
    __ultimoErro = None

    def __init__(self,nomeAmbiente:str,classe:int,posicao:str,repeticao:int,grupo:int,caminhoArquivo:str,frequencias:np.ndarray,parteRealS21:np.ndarray,parteImaginariaS21:np.ndarray):
        self.__nomeAmbiente = nomeAmbiente
        self.__classe = classe
        self.__posicao = posicao
        self.__repeticao = repeticao
        self.__grupo = grupo
        self.__caminhoArquivo = caminhoArquivo
        self.__frequencias = np.asarray(frequencias,dtype=np.float64).copy()
        self.__parteRealS21 = np.asarray(parteRealS21,dtype=np.float64).copy()
        self.__parteImaginariaS21 = np.asarray(parteImaginariaS21,dtype=np.float64).copy()

    @staticmethod
    def criar(nomeAmbiente:str,classe:int,posicao:str,repeticao:int,grupo:int,caminhoArquivo:str,frequencias:np.ndarray,parteRealS21:np.ndarray,parteImaginariaS21:np.ndarray) -> MedicaoCanal | None:
        MedicaoCanal.__limparUltimoErro()

        try:
            medicao = MedicaoCanal(nomeAmbiente,classe,posicao,repeticao,grupo,caminhoArquivo,frequencias,parteRealS21,parteImaginariaS21)

            if(medicao.validar()):
                return medicao

            else:
                return None

        except Exception as excecao:
            MedicaoCanal.__registrarErro(CODIGO_ERRO_CRIACAO_MEDICAO,"Não foi possível criar a medição","MedicaoCanal.criar",str(excecao))
            return None

    @staticmethod
    def __registrarErro(codigo:int,mensagem:str,origem:str,detalhe:str | None=None) -> None:
        MedicaoCanal.__ultimoErro = ErroSistema.criar(codigo,mensagem,origem,detalhe)

    @staticmethod
    def __limparUltimoErro() -> None:
        MedicaoCanal.__ultimoErro = None

    @staticmethod
    def getUltimoErro() -> ErroSistema | None:
        return MedicaoCanal.__ultimoErro

    @staticmethod
    def ultimaExecucaoDeuErro() -> bool:
        if(MedicaoCanal.__ultimoErro is None):
            return False

        else:
            return True

    def validar(self) -> bool:
        MedicaoCanal.__limparUltimoErro()

        try:
            frequenciasEsperadas = np.linspace(FREQUENCIA_INICIAL_HZ,FREQUENCIA_FINAL_HZ,QUANTIDADE_PONTOS_FREQUENCIA)

            if(not isinstance(self.__nomeAmbiente,str) or self.__nomeAmbiente.strip() == ""):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_NOME_AMBIENTE_INVALIDO,"O nome do ambiente é inválido","MedicaoCanal.validar")
                return False

            elif(not isinstance(self.__classe,int) or isinstance(self.__classe,bool)):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_CLASSE_INVALIDA,"A classe do ambiente é inválida","MedicaoCanal.validar")
                return False

            elif(self.__classe < 0 or self.__classe >= QUANTIDADE_AMBIENTES):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_CLASSE_INVALIDA,"A classe do ambiente está fora do intervalo esperado","MedicaoCanal.validar")
                return False

            elif(not isinstance(self.__posicao,str) or self.__posicao.strip() == ""):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_POSICAO_INVALIDA,"A posição da medição é inválida","MedicaoCanal.validar")
                return False

            elif(not isinstance(self.__repeticao,int) or isinstance(self.__repeticao,bool)):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_REPETICAO_INVALIDA,"O número da repetição é inválido","MedicaoCanal.validar")
                return False

            elif(self.__repeticao < 1 or self.__repeticao > QUANTIDADE_REPETICOES_POR_POSICAO):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_REPETICAO_INVALIDA,"O número da repetição está fora do intervalo esperado","MedicaoCanal.validar")
                return False

            elif(not isinstance(self.__grupo,int) or isinstance(self.__grupo,bool)):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_GRUPO_INVALIDO,"O grupo da posição é inválido","MedicaoCanal.validar")
                return False

            elif(self.__grupo < 0 or self.__grupo >= QUANTIDADE_GRUPOS_TOTAL):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_GRUPO_INVALIDO,"O grupo da posição está fora do intervalo esperado","MedicaoCanal.validar")
                return False

            elif(not isinstance(self.__caminhoArquivo,str) or self.__caminhoArquivo.strip() == ""):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_CAMINHO_MEDICAO_INVALIDO,"O caminho do arquivo de medição é inválido","MedicaoCanal.validar")
                return False

            elif(self.__frequencias.ndim != 1):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_DIMENSAO_INVALIDA,"O vetor de frequências deve possuir uma dimensão","MedicaoCanal.validar")
                return False

            elif(self.__parteRealS21.ndim != 1 or self.__parteImaginariaS21.ndim != 1):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_DIMENSAO_INVALIDA,"As componentes de S21 devem possuir uma dimensão","MedicaoCanal.validar")
                return False

            elif(self.__frequencias.size != QUANTIDADE_PONTOS_FREQUENCIA):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_QUANTIDADE_PONTOS_INVALIDA,"A quantidade de frequências é inválida","MedicaoCanal.validar")
                return False

            elif(self.__parteRealS21.size != QUANTIDADE_PONTOS_FREQUENCIA or self.__parteImaginariaS21.size != QUANTIDADE_PONTOS_FREQUENCIA):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_QUANTIDADE_PONTOS_INVALIDA,"A quantidade de pontos de S21 é inválida","MedicaoCanal.validar")
                return False

            elif(not np.all(np.isfinite(self.__frequencias))):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_VALOR_NUMERICO_INVALIDO,"O vetor de frequências contém valores inválidos","MedicaoCanal.validar")
                return False

            elif(not np.all(np.isfinite(self.__parteRealS21)) or not np.all(np.isfinite(self.__parteImaginariaS21))):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_VALOR_NUMERICO_INVALIDO,"S21 contém valores inválidos","MedicaoCanal.validar")
                return False

            elif(not np.allclose(self.__frequencias,frequenciasEsperadas,rtol=0.0,atol=TOLERANCIA_FREQUENCIA_HZ)):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_GRADE_FREQUENCIAS_INVALIDA,"A grade de frequências é inconsistente","MedicaoCanal.validar")
                return False

            else:
                MedicaoCanal.__limparUltimoErro()
                return True

        except Exception as excecao:
            MedicaoCanal.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado durante a validação da medição","MedicaoCanal.validar",str(excecao))
            return False

    def getNomeAmbiente(self) -> str:
        return self.__nomeAmbiente

    def getClasse(self) -> int:
        return self.__classe

    def getPosicao(self) -> str:
        return self.__posicao

    def getRepeticao(self) -> int:
        return self.__repeticao

    def getGrupo(self) -> int:
        return self.__grupo

    def getCaminhoArquivo(self) -> str:
        return self.__caminhoArquivo

    def getFrequencias(self) -> np.ndarray:
        return self.__frequencias.copy()

    def getParteRealS21(self) -> np.ndarray:
        return self.__parteRealS21.copy()

    def getParteImaginariaS21(self) -> np.ndarray:
        return self.__parteImaginariaS21.copy()

    def setNomeAmbiente(self,nomeAmbiente:str) -> bool:
        MedicaoCanal.__limparUltimoErro()

        try:
            if(not isinstance(nomeAmbiente,str) or nomeAmbiente.strip() == ""):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_NOME_AMBIENTE_INVALIDO,"O nome do ambiente é inválido","MedicaoCanal.setNomeAmbiente")
                return False

            else:
                self.__nomeAmbiente = nomeAmbiente
                return True

        except Exception as excecao:
            MedicaoCanal.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado ao alterar o nome do ambiente","MedicaoCanal.setNomeAmbiente",str(excecao))
            return False

    def setClasse(self,classe:int) -> bool:
        MedicaoCanal.__limparUltimoErro()

        try:
            if(not isinstance(classe,int) or isinstance(classe,bool)):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_CLASSE_INVALIDA,"A classe do ambiente é inválida","MedicaoCanal.setClasse")
                return False

            elif(classe < 0 or classe >= QUANTIDADE_AMBIENTES):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_CLASSE_INVALIDA,"A classe do ambiente está fora do intervalo esperado","MedicaoCanal.setClasse")
                return False

            else:
                self.__classe = classe
                return True

        except Exception as excecao:
            MedicaoCanal.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado ao alterar a classe","MedicaoCanal.setClasse",str(excecao))
            return False

    def setPosicao(self,posicao:str) -> bool:
        MedicaoCanal.__limparUltimoErro()

        try:
            if(not isinstance(posicao,str) or posicao.strip() == ""):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_POSICAO_INVALIDA,"A posição da medição é inválida","MedicaoCanal.setPosicao")
                return False

            else:
                self.__posicao = posicao
                return True

        except Exception as excecao:
            MedicaoCanal.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado ao alterar a posição","MedicaoCanal.setPosicao",str(excecao))
            return False

    def setRepeticao(self,repeticao:int) -> bool:
        MedicaoCanal.__limparUltimoErro()

        try:
            if(not isinstance(repeticao,int) or isinstance(repeticao,bool)):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_REPETICAO_INVALIDA,"O número da repetição é inválido","MedicaoCanal.setRepeticao")
                return False

            elif(repeticao < 1 or repeticao > QUANTIDADE_REPETICOES_POR_POSICAO):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_REPETICAO_INVALIDA,"O número da repetição está fora do intervalo esperado","MedicaoCanal.setRepeticao")
                return False

            else:
                self.__repeticao = repeticao
                return True

        except Exception as excecao:
            MedicaoCanal.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado ao alterar a repetição","MedicaoCanal.setRepeticao",str(excecao))
            return False

    def setGrupo(self,grupo:int) -> bool:
        MedicaoCanal.__limparUltimoErro()

        try:
            if(not isinstance(grupo,int) or isinstance(grupo,bool)):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_GRUPO_INVALIDO,"O grupo da posição é inválido","MedicaoCanal.setGrupo")
                return False

            elif(grupo < 0 or grupo >= QUANTIDADE_GRUPOS_TOTAL):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_GRUPO_INVALIDO,"O grupo da posição está fora do intervalo esperado","MedicaoCanal.setGrupo")
                return False

            else:
                self.__grupo = grupo
                return True

        except Exception as excecao:
            MedicaoCanal.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado ao alterar o grupo","MedicaoCanal.setGrupo",str(excecao))
            return False

    def setCaminhoArquivo(self,caminhoArquivo:str) -> bool:
        MedicaoCanal.__limparUltimoErro()

        try:
            if(not isinstance(caminhoArquivo,str) or caminhoArquivo.strip() == ""):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_CAMINHO_MEDICAO_INVALIDO,"O caminho do arquivo de medição é inválido","MedicaoCanal.setCaminhoArquivo")
                return False

            else:
                self.__caminhoArquivo = caminhoArquivo
                return True

        except Exception as excecao:
            MedicaoCanal.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado ao alterar o caminho do arquivo","MedicaoCanal.setCaminhoArquivo",str(excecao))
            return False

    def setFrequencias(self,frequencias:np.ndarray) -> bool:
        MedicaoCanal.__limparUltimoErro()

        try:
            novoVetor = np.asarray(frequencias,dtype=np.float64)
            frequenciasEsperadas = np.linspace(FREQUENCIA_INICIAL_HZ,FREQUENCIA_FINAL_HZ,QUANTIDADE_PONTOS_FREQUENCIA)

            if(novoVetor.ndim != 1):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_DIMENSAO_INVALIDA,"O vetor de frequências deve possuir uma dimensão","MedicaoCanal.setFrequencias")
                return False

            elif(novoVetor.size != QUANTIDADE_PONTOS_FREQUENCIA):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_QUANTIDADE_PONTOS_INVALIDA,"A quantidade de frequências é inválida","MedicaoCanal.setFrequencias")
                return False

            elif(not np.all(np.isfinite(novoVetor))):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_VALOR_NUMERICO_INVALIDO,"O vetor de frequências contém valores inválidos","MedicaoCanal.setFrequencias")
                return False

            elif(not np.allclose(novoVetor,frequenciasEsperadas,rtol=0.0,atol=TOLERANCIA_FREQUENCIA_HZ)):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_GRADE_FREQUENCIAS_INVALIDA,"A grade de frequências é inconsistente","MedicaoCanal.setFrequencias")
                return False

            else:
                self.__frequencias = novoVetor.copy()
                return True

        except Exception as excecao:
            MedicaoCanal.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado ao alterar as frequências","MedicaoCanal.setFrequencias",str(excecao))
            return False

    def setParteRealS21(self,parteRealS21:np.ndarray) -> bool:
        MedicaoCanal.__limparUltimoErro()

        try:
            novoVetor = np.asarray(parteRealS21,dtype=np.float64)

            if(novoVetor.ndim != 1):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_DIMENSAO_INVALIDA,"A parte real de S21 deve possuir uma dimensão","MedicaoCanal.setParteRealS21")
                return False

            elif(novoVetor.size != QUANTIDADE_PONTOS_FREQUENCIA):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_QUANTIDADE_PONTOS_INVALIDA,"A quantidade de pontos da parte real de S21 é inválida","MedicaoCanal.setParteRealS21")
                return False

            elif(not np.all(np.isfinite(novoVetor))):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_VALOR_NUMERICO_INVALIDO,"A parte real de S21 contém valores inválidos","MedicaoCanal.setParteRealS21")
                return False

            else:
                self.__parteRealS21 = novoVetor.copy()
                return True

        except Exception as excecao:
            MedicaoCanal.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado ao alterar a parte real de S21","MedicaoCanal.setParteRealS21",str(excecao))
            return False

    def setParteImaginariaS21(self,parteImaginariaS21:np.ndarray) -> bool:
        MedicaoCanal.__limparUltimoErro()

        try:
            novoVetor = np.asarray(parteImaginariaS21,dtype=np.float64)

            if(novoVetor.ndim != 1):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_DIMENSAO_INVALIDA,"A parte imaginária de S21 deve possuir uma dimensão","MedicaoCanal.setParteImaginariaS21")
                return False

            elif(novoVetor.size != QUANTIDADE_PONTOS_FREQUENCIA):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_QUANTIDADE_PONTOS_INVALIDA,"A quantidade de pontos da parte imaginária de S21 é inválida","MedicaoCanal.setParteImaginariaS21")
                return False

            elif(not np.all(np.isfinite(novoVetor))):
                MedicaoCanal.__registrarErro(CODIGO_ERRO_VALOR_NUMERICO_INVALIDO,"A parte imaginária de S21 contém valores inválidos","MedicaoCanal.setParteImaginariaS21")
                return False

            else:
                self.__parteImaginariaS21 = novoVetor.copy()
                return True

        except Exception as excecao:
            MedicaoCanal.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado ao alterar a parte imaginária de S21","MedicaoCanal.setParteImaginariaS21",str(excecao))
            return False

    def obterS21Complexo(self) -> np.ndarray | None:
        MedicaoCanal.__limparUltimoErro()

        try:
            return self.__parteRealS21 + 1j * self.__parteImaginariaS21

        except Exception as excecao:
            MedicaoCanal.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado ao construir S21 complexo","MedicaoCanal.obterS21Complexo",str(excecao))
            return None

    def obterS21EmDoisCanais(self) -> np.ndarray | None:
        MedicaoCanal.__limparUltimoErro()

        try:
            return np.column_stack((self.__parteRealS21,self.__parteImaginariaS21))

        except Exception as excecao:
            MedicaoCanal.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado ao construir os dois canais de S21","MedicaoCanal.obterS21EmDoisCanais",str(excecao))
            return None