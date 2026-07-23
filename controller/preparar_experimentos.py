import sys

import numpy as np

from model.entidades.erro_sistema import ErroSistema
from services.constantes import CAMINHO_BASE_PREPARADA
from services.constantes import CHAVE_CLASSES
from services.constantes import CHAVE_GRUPOS
from services.constantes import CHAVE_INDICES_TESTE
from services.constantes import CHAVE_INDICES_TREINO
from services.constantes import CHAVE_INDICES_VALIDACAO
from services.constantes import CHAVE_REPRESENTACAO_CARTESIANA
from services.constantes import CHAVE_REPRESENTACAO_POLAR
from services.constantes import CHAVE_REPRESENTACAO_TEMPORAL
from services.constantes import QUANTIDADE_AMBIENTES
from services.preparador_experimentos import PreparadorExperimentos


def exibirErro(erro:ErroSistema | None) -> None:
    try:
        if(erro is None):
            print("Ocorreu um erro sem informações adicionais.")

        else:
            descricao = erro.obterDescricaoCompleta()

            if(descricao is None):
                print("Ocorreu um erro, mas não foi possível montar sua descrição.")

            else:
                print(descricao)

    except Exception as excecao:
        print("Não foi possível exibir o erro: " + str(excecao))


def executar() -> bool:
    try:
        print("Iniciando a preparação dos experimentos.")

        if(not PreparadorExperimentos.prepararESalvar()):
            exibirErro(PreparadorExperimentos.getUltimoErro())
            return False

        else:
            dicionarioBase = PreparadorExperimentos.carregarBasePreparada()

            if(dicionarioBase is None):
                exibirErro(PreparadorExperimentos.getUltimoErro())
                return False

            else:
                vetorClasses = dicionarioBase[CHAVE_CLASSES]
                vetorGrupos = dicionarioBase[CHAVE_GRUPOS]

                indicesTreino = dicionarioBase[CHAVE_INDICES_TREINO]
                indicesValidacao = dicionarioBase[CHAVE_INDICES_VALIDACAO]
                indicesTeste = dicionarioBase[CHAVE_INDICES_TESTE]

                classesTreino = np.bincount(vetorClasses[indicesTreino].astype(np.int64),minlength=QUANTIDADE_AMBIENTES)
                classesValidacao = np.bincount(vetorClasses[indicesValidacao].astype(np.int64),minlength=QUANTIDADE_AMBIENTES)
                classesTeste = np.bincount(vetorClasses[indicesTeste].astype(np.int64),minlength=QUANTIDADE_AMBIENTES)

                quantidadeGruposTreino = np.unique(vetorGrupos[indicesTreino]).size
                quantidadeGruposValidacao = np.unique(vetorGrupos[indicesValidacao]).size
                quantidadeGruposTeste = np.unique(vetorGrupos[indicesTeste]).size

                print("Base preparada salva em: " + str(CAMINHO_BASE_PREPARADA))
                print("Representação cartesiana: " + str(dicionarioBase[CHAVE_REPRESENTACAO_CARTESIANA].shape))
                print("Representação polar: " + str(dicionarioBase[CHAVE_REPRESENTACAO_POLAR].shape))
                print("Representação temporal: " + str(dicionarioBase[CHAVE_REPRESENTACAO_TEMPORAL].shape))
                print("Medições de treino: " + str(indicesTreino.size))
                print("Medições de validação: " + str(indicesValidacao.size))
                print("Medições de teste: " + str(indicesTeste.size))
                print("Classes no treino: " + str(classesTreino.tolist()))
                print("Classes na validação: " + str(classesValidacao.tolist()))
                print("Classes no teste: " + str(classesTeste.tolist()))
                print("Grupos de treino: " + str(quantidadeGruposTreino))
                print("Grupos de validação: " + str(quantidadeGruposValidacao))
                print("Grupos de teste: " + str(quantidadeGruposTeste))
                print("Tamanho do arquivo: %.2f MB" % (CAMINHO_BASE_PREPARADA.stat().st_size / 1048576.0))

                return True

    except Exception as excecao:
        print("Erro inesperado no controlador: " + str(excecao))
        return False


if(__name__ == "__main__"):
    if(executar()):
        sys.exit(0)

    else:
        sys.exit(1)