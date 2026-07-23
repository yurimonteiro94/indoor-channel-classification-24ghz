from __future__ import annotations

import numpy as np

from model.entidades.erro_sistema import ErroSistema
from services.constantes import CHAVE_GRUPOS_TESTE
from services.constantes import CHAVE_GRUPOS_TREINO
from services.constantes import CHAVE_GRUPOS_VALIDACAO
from services.constantes import CHAVE_INDICES_TESTE
from services.constantes import CHAVE_INDICES_TREINO
from services.constantes import CHAVE_INDICES_VALIDACAO
from services.constantes import CODIGO_ERRO_CRIACAO_DIVISAO
from services.constantes import CODIGO_ERRO_DESCONHECIDO
from services.constantes import CODIGO_ERRO_DIVISAO_INVALIDA
from services.constantes import CODIGO_ERRO_GRUPO_COM_MULTIPLAS_CLASSES
from services.constantes import CODIGO_ERRO_GRUPOS_INSUFICIENTES
from services.constantes import CODIGO_ERRO_VETORES_DIVISAO_INVALIDOS
from services.constantes import PROPORCAO_TREINO
from services.constantes import PROPORCAO_VALIDACAO
from services.constantes import QUANTIDADE_AMBIENTES
from services.constantes import QUANTIDADE_GRUPOS_TOTAL
from services.constantes import SEMENTE_ALEATORIA


class SeparadorBase:
    __ultimoErro = None

    @staticmethod
    def __registrarErro(codigo:int,mensagem:str,origem:str,detalhe:str | None=None) -> None:
        SeparadorBase.__ultimoErro = ErroSistema.criar(codigo,mensagem,origem,detalhe)

    @staticmethod
    def __limparUltimoErro() -> None:
        SeparadorBase.__ultimoErro = None

    @staticmethod
    def getUltimoErro() -> ErroSistema | None:
        return SeparadorBase.__ultimoErro

    @staticmethod
    def ultimaExecucaoDeuErro() -> bool:
        if(SeparadorBase.__ultimoErro is None):
            return False

        else:
            return True

    @staticmethod
    def __validarVetores(vetorClasses:np.ndarray,vetorGrupos:np.ndarray) -> bool:
        try:
            if(vetorClasses.ndim != 1):
                SeparadorBase.__registrarErro(CODIGO_ERRO_VETORES_DIVISAO_INVALIDOS,"O vetor de classes deve possuir uma dimensão","SeparadorBase.__validarVetores")
                return False

            elif(vetorGrupos.ndim != 1):
                SeparadorBase.__registrarErro(CODIGO_ERRO_VETORES_DIVISAO_INVALIDOS,"O vetor de grupos deve possuir uma dimensão","SeparadorBase.__validarVetores")
                return False

            elif(vetorClasses.size == 0):
                SeparadorBase.__registrarErro(CODIGO_ERRO_VETORES_DIVISAO_INVALIDOS,"O vetor de classes não pode ser vazio","SeparadorBase.__validarVetores")
                return False

            elif(vetorClasses.size != vetorGrupos.size):
                SeparadorBase.__registrarErro(CODIGO_ERRO_VETORES_DIVISAO_INVALIDOS,"Os vetores de classes e grupos devem possuir o mesmo tamanho","SeparadorBase.__validarVetores")
                return False

            elif(not np.issubdtype(vetorClasses.dtype,np.integer)):
                SeparadorBase.__registrarErro(CODIGO_ERRO_VETORES_DIVISAO_INVALIDOS,"O vetor de classes deve possuir valores inteiros","SeparadorBase.__validarVetores")
                return False

            elif(not np.issubdtype(vetorGrupos.dtype,np.integer)):
                SeparadorBase.__registrarErro(CODIGO_ERRO_VETORES_DIVISAO_INVALIDOS,"O vetor de grupos deve possuir valores inteiros","SeparadorBase.__validarVetores")
                return False

            elif(np.any(vetorClasses < 0) or np.any(vetorClasses >= QUANTIDADE_AMBIENTES)):
                SeparadorBase.__registrarErro(CODIGO_ERRO_VETORES_DIVISAO_INVALIDOS,"O vetor de classes contém valores inválidos","SeparadorBase.__validarVetores")
                return False

            elif(np.any(vetorGrupos < 0) or np.any(vetorGrupos >= QUANTIDADE_GRUPOS_TOTAL)):
                SeparadorBase.__registrarErro(CODIGO_ERRO_VETORES_DIVISAO_INVALIDOS,"O vetor de grupos contém valores inválidos","SeparadorBase.__validarVetores")
                return False

            else:
                gruposUnicos = np.unique(vetorGrupos)

                for grupo in gruposUnicos:
                    mascaraGrupo = vetorGrupos == grupo
                    classesDoGrupo = np.unique(vetorClasses[mascaraGrupo])

                    if(classesDoGrupo.size != 1):
                        SeparadorBase.__registrarErro(CODIGO_ERRO_GRUPO_COM_MULTIPLAS_CLASSES,"Um grupo físico está associado a mais de uma classe","SeparadorBase.__validarVetores",str(int(grupo)))
                        return False

                return True

        except Exception as excecao:
            SeparadorBase.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado durante a validação dos vetores","SeparadorBase.__validarVetores",str(excecao))
            return False

    @staticmethod
    def criarDivisao(vetorClasses:np.ndarray,vetorGrupos:np.ndarray,semente:int=SEMENTE_ALEATORIA) -> dict[str,np.ndarray] | None:
        SeparadorBase.__limparUltimoErro()

        try:
            vetorClasses = np.asarray(vetorClasses)
            vetorGrupos = np.asarray(vetorGrupos)

            if(not SeparadorBase.__validarVetores(vetorClasses,vetorGrupos)):
                return None

            elif(not isinstance(semente,int) or isinstance(semente,bool)):
                SeparadorBase.__registrarErro(CODIGO_ERRO_CRIACAO_DIVISAO,"A semente aleatória é inválida","SeparadorBase.criarDivisao")
                return None

            else:
                geradorAleatorio = np.random.default_rng(semente)

                listaDeGruposTreino = []
                listaDeGruposValidacao = []
                listaDeGruposTeste = []

                for classe in range(QUANTIDADE_AMBIENTES):
                    mascaraClasse = vetorClasses == classe
                    gruposDaClasse = np.unique(vetorGrupos[mascaraClasse])
                    quantidadeDeGrupos = gruposDaClasse.size

                    if(quantidadeDeGrupos < 3):
                        SeparadorBase.__registrarErro(CODIGO_ERRO_GRUPOS_INSUFICIENTES,"Uma classe não possui grupos suficientes para os três subconjuntos","SeparadorBase.criarDivisao",str(classe))
                        return None

                    else:
                        gruposEmbaralhados = geradorAleatorio.permutation(gruposDaClasse)

                        quantidadeTreino = int(np.floor(quantidadeDeGrupos * PROPORCAO_TREINO))
                        quantidadeValidacao = int(np.floor(quantidadeDeGrupos * PROPORCAO_VALIDACAO))

                        if(quantidadeTreino < 1):
                            quantidadeTreino = 1

                        if(quantidadeValidacao < 1):
                            quantidadeValidacao = 1

                        if(quantidadeTreino + quantidadeValidacao >= quantidadeDeGrupos):
                            quantidadeTreino = quantidadeDeGrupos - 2
                            quantidadeValidacao = 1

                        quantidadeTeste = quantidadeDeGrupos - quantidadeTreino - quantidadeValidacao

                        if(quantidadeTeste < 1):
                            SeparadorBase.__registrarErro(CODIGO_ERRO_GRUPOS_INSUFICIENTES,"Não foi possível reservar grupos para o teste","SeparadorBase.criarDivisao",str(classe))
                            return None

                        else:
                            for indiceGrupo in range(quantidadeTreino):
                                listaDeGruposTreino.append(int(gruposEmbaralhados[indiceGrupo]))

                            limiteValidacao = quantidadeTreino + quantidadeValidacao

                            for indiceGrupo in range(quantidadeTreino,limiteValidacao):
                                listaDeGruposValidacao.append(int(gruposEmbaralhados[indiceGrupo]))

                            for indiceGrupo in range(limiteValidacao,quantidadeDeGrupos):
                                listaDeGruposTeste.append(int(gruposEmbaralhados[indiceGrupo]))

                vetorGruposTreino = np.asarray(listaDeGruposTreino,dtype=np.int16)
                vetorGruposValidacao = np.asarray(listaDeGruposValidacao,dtype=np.int16)
                vetorGruposTeste = np.asarray(listaDeGruposTeste,dtype=np.int16)

                conjuntoGruposTreino = set(vetorGruposTreino.tolist())
                conjuntoGruposValidacao = set(vetorGruposValidacao.tolist())
                conjuntoGruposTeste = set(vetorGruposTeste.tolist())

                listaDeIndicesTreino = []
                listaDeIndicesValidacao = []
                listaDeIndicesTeste = []

                for indiceMedicao in range(vetorGrupos.size):
                    grupo = int(vetorGrupos[indiceMedicao])

                    if(grupo in conjuntoGruposTreino):
                        listaDeIndicesTreino.append(indiceMedicao)

                    elif(grupo in conjuntoGruposValidacao):
                        listaDeIndicesValidacao.append(indiceMedicao)

                    elif(grupo in conjuntoGruposTeste):
                        listaDeIndicesTeste.append(indiceMedicao)

                    else:
                        SeparadorBase.__registrarErro(CODIGO_ERRO_DIVISAO_INVALIDA,"Uma medição não foi associada a nenhum subconjunto","SeparadorBase.criarDivisao",str(indiceMedicao))
                        return None

                vetorIndicesTreino = np.asarray(listaDeIndicesTreino,dtype=np.int64)
                vetorIndicesValidacao = np.asarray(listaDeIndicesValidacao,dtype=np.int64)
                vetorIndicesTeste = np.asarray(listaDeIndicesTeste,dtype=np.int64)

                vetorIndicesTreino = geradorAleatorio.permutation(vetorIndicesTreino)
                vetorIndicesValidacao = geradorAleatorio.permutation(vetorIndicesValidacao)
                vetorIndicesTeste = geradorAleatorio.permutation(vetorIndicesTeste)

                dicionarioDivisao = {
                    CHAVE_INDICES_TREINO: vetorIndicesTreino,
                    CHAVE_INDICES_VALIDACAO: vetorIndicesValidacao,
                    CHAVE_INDICES_TESTE: vetorIndicesTeste,
                    CHAVE_GRUPOS_TREINO: vetorGruposTreino,
                    CHAVE_GRUPOS_VALIDACAO: vetorGruposValidacao,
                    CHAVE_GRUPOS_TESTE: vetorGruposTeste
                }

                if(not SeparadorBase.validarDivisao(dicionarioDivisao,vetorClasses,vetorGrupos)):
                    return None

                else:
                    return dicionarioDivisao

        except Exception as excecao:
            SeparadorBase.__registrarErro(CODIGO_ERRO_CRIACAO_DIVISAO,"Não foi possível criar a divisão da base","SeparadorBase.criarDivisao",str(excecao))
            return None

    @staticmethod
    def validarDivisao(dicionarioDivisao:dict[str,np.ndarray],vetorClasses:np.ndarray,vetorGrupos:np.ndarray) -> bool:
        SeparadorBase.__limparUltimoErro()

        try:
            vetorClasses = np.asarray(vetorClasses)
            vetorGrupos = np.asarray(vetorGrupos)

            if(not SeparadorBase.__validarVetores(vetorClasses,vetorGrupos)):
                return False

            elif(not isinstance(dicionarioDivisao,dict)):
                SeparadorBase.__registrarErro(CODIGO_ERRO_DIVISAO_INVALIDA,"A divisão informada não é um dicionário","SeparadorBase.validarDivisao")
                return False

            else:
                chavesObrigatorias = (CHAVE_INDICES_TREINO,CHAVE_INDICES_VALIDACAO,CHAVE_INDICES_TESTE,CHAVE_GRUPOS_TREINO,CHAVE_GRUPOS_VALIDACAO,CHAVE_GRUPOS_TESTE)

                for chave in chavesObrigatorias:
                    if(chave not in dicionarioDivisao):
                        SeparadorBase.__registrarErro(CODIGO_ERRO_DIVISAO_INVALIDA,"Uma chave obrigatória não foi encontrada na divisão","SeparadorBase.validarDivisao",chave)
                        return False

                vetorIndicesTreino = np.asarray(dicionarioDivisao[CHAVE_INDICES_TREINO])
                vetorIndicesValidacao = np.asarray(dicionarioDivisao[CHAVE_INDICES_VALIDACAO])
                vetorIndicesTeste = np.asarray(dicionarioDivisao[CHAVE_INDICES_TESTE])
                vetorGruposTreino = np.asarray(dicionarioDivisao[CHAVE_GRUPOS_TREINO])
                vetorGruposValidacao = np.asarray(dicionarioDivisao[CHAVE_GRUPOS_VALIDACAO])
                vetorGruposTeste = np.asarray(dicionarioDivisao[CHAVE_GRUPOS_TESTE])

                listaDeVetores = [
                    ("índices de treino",vetorIndicesTreino),
                    ("índices de validação",vetorIndicesValidacao),
                    ("índices de teste",vetorIndicesTeste),
                    ("grupos de treino",vetorGruposTreino),
                    ("grupos de validação",vetorGruposValidacao),
                    ("grupos de teste",vetorGruposTeste)
                ]

                for item in listaDeVetores:
                    nomeVetor = item[0]
                    vetor = item[1]

                    if(vetor.ndim != 1):
                        SeparadorBase.__registrarErro(CODIGO_ERRO_DIVISAO_INVALIDA,"Um vetor da divisão não possui uma dimensão","SeparadorBase.validarDivisao",nomeVetor)
                        return False

                    elif(vetor.size == 0):
                        SeparadorBase.__registrarErro(CODIGO_ERRO_DIVISAO_INVALIDA,"Um vetor da divisão está vazio","SeparadorBase.validarDivisao",nomeVetor)
                        return False

                    elif(not np.issubdtype(vetor.dtype,np.integer)):
                        SeparadorBase.__registrarErro(CODIGO_ERRO_DIVISAO_INVALIDA,"Um vetor da divisão não possui valores inteiros","SeparadorBase.validarDivisao",nomeVetor)
                        return False

                quantidadeDeMedicoes = vetorClasses.size

                if(np.any(vetorIndicesTreino < 0) or np.any(vetorIndicesTreino >= quantidadeDeMedicoes)):
                    SeparadorBase.__registrarErro(CODIGO_ERRO_DIVISAO_INVALIDA,"O vetor de índices de treino contém valores inválidos","SeparadorBase.validarDivisao")
                    return False

                elif(np.any(vetorIndicesValidacao < 0) or np.any(vetorIndicesValidacao >= quantidadeDeMedicoes)):
                    SeparadorBase.__registrarErro(CODIGO_ERRO_DIVISAO_INVALIDA,"O vetor de índices de validação contém valores inválidos","SeparadorBase.validarDivisao")
                    return False

                elif(np.any(vetorIndicesTeste < 0) or np.any(vetorIndicesTeste >= quantidadeDeMedicoes)):
                    SeparadorBase.__registrarErro(CODIGO_ERRO_DIVISAO_INVALIDA,"O vetor de índices de teste contém valores inválidos","SeparadorBase.validarDivisao")
                    return False

                else:
                    todosIndices = np.concatenate((vetorIndicesTreino,vetorIndicesValidacao,vetorIndicesTeste))
                    todosGrupos = np.concatenate((vetorGruposTreino,vetorGruposValidacao,vetorGruposTeste))
                    gruposUnicosDaBase = np.unique(vetorGrupos)

                    if(todosIndices.size != quantidadeDeMedicoes):
                        SeparadorBase.__registrarErro(CODIGO_ERRO_DIVISAO_INVALIDA,"A divisão não contém todas as medições","SeparadorBase.validarDivisao")
                        return False

                    elif(np.unique(todosIndices).size != quantidadeDeMedicoes):
                        SeparadorBase.__registrarErro(CODIGO_ERRO_DIVISAO_INVALIDA,"Existem medições repetidas entre os subconjuntos","SeparadorBase.validarDivisao")
                        return False

                    elif(not np.array_equal(np.sort(todosIndices),np.arange(quantidadeDeMedicoes,dtype=np.int64))):
                        SeparadorBase.__registrarErro(CODIGO_ERRO_DIVISAO_INVALIDA,"Os índices da divisão são inconsistentes","SeparadorBase.validarDivisao")
                        return False

                    elif(np.unique(todosGrupos).size != todosGrupos.size):
                        SeparadorBase.__registrarErro(CODIGO_ERRO_DIVISAO_INVALIDA,"Existem grupos repetidos entre os subconjuntos","SeparadorBase.validarDivisao")
                        return False

                    elif(np.unique(todosGrupos).size != gruposUnicosDaBase.size):
                        SeparadorBase.__registrarErro(CODIGO_ERRO_DIVISAO_INVALIDA,"A divisão não contém todos os grupos físicos","SeparadorBase.validarDivisao")
                        return False

                    elif(not np.array_equal(np.sort(np.unique(vetorGrupos[vetorIndicesTreino])),np.sort(vetorGruposTreino))):
                        SeparadorBase.__registrarErro(CODIGO_ERRO_DIVISAO_INVALIDA,"Os grupos declarados para o treino são inconsistentes","SeparadorBase.validarDivisao")
                        return False

                    elif(not np.array_equal(np.sort(np.unique(vetorGrupos[vetorIndicesValidacao])),np.sort(vetorGruposValidacao))):
                        SeparadorBase.__registrarErro(CODIGO_ERRO_DIVISAO_INVALIDA,"Os grupos declarados para a validação são inconsistentes","SeparadorBase.validarDivisao")
                        return False

                    elif(not np.array_equal(np.sort(np.unique(vetorGrupos[vetorIndicesTeste])),np.sort(vetorGruposTeste))):
                        SeparadorBase.__registrarErro(CODIGO_ERRO_DIVISAO_INVALIDA,"Os grupos declarados para o teste são inconsistentes","SeparadorBase.validarDivisao")
                        return False

                    else:
                        for classe in range(QUANTIDADE_AMBIENTES):
                            if(not np.any(vetorClasses[vetorIndicesTreino] == classe)):
                                SeparadorBase.__registrarErro(CODIGO_ERRO_DIVISAO_INVALIDA,"Uma classe não está presente no treino","SeparadorBase.validarDivisao",str(classe))
                                return False

                            elif(not np.any(vetorClasses[vetorIndicesValidacao] == classe)):
                                SeparadorBase.__registrarErro(CODIGO_ERRO_DIVISAO_INVALIDA,"Uma classe não está presente na validação","SeparadorBase.validarDivisao",str(classe))
                                return False

                            elif(not np.any(vetorClasses[vetorIndicesTeste] == classe)):
                                SeparadorBase.__registrarErro(CODIGO_ERRO_DIVISAO_INVALIDA,"Uma classe não está presente no teste","SeparadorBase.validarDivisao",str(classe))
                                return False

                        return True

        except Exception as excecao:
            SeparadorBase.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado durante a validação da divisão","SeparadorBase.validarDivisao",str(excecao))
            return False