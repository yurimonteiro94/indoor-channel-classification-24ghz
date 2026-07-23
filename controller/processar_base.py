import numpy as np

from model.entidades.erro_sistema import ErroSistema
from services.constantes import CAMINHO_BASE_PROCESSADA
from services.constantes import CHAVE_CLASSES
from services.constantes import CHAVE_GRUPOS
from services.constantes import CHAVE_S21
from services.constantes import QUANTIDADE_AMBIENTES
from services.processador_base import ProcessadorBase


def exibirErro(erro:ErroSistema | None) -> None:
    if(erro is None):
        print("Ocorreu um erro sem informações adicionais.")

    else:
        descricao = erro.obterDescricaoCompleta()

        if(descricao is None):
            print("Ocorreu um erro, mas não foi possível montar sua descrição.")

        else:
            print(descricao)


def executar() -> bool:
    try:
        print("Iniciando o processamento da base.")

        if(not ProcessadorBase.processarESalvar()):
            exibirErro(ProcessadorBase.getUltimoErro())
            return False

        else:
            dicionarioBase = ProcessadorBase.carregarBase(CAMINHO_BASE_PROCESSADA,True)

            if(dicionarioBase is None):
                exibirErro(ProcessadorBase.getUltimoErro())
                return False

            else:
                matrizS21 = dicionarioBase[CHAVE_S21]
                vetorClasses = dicionarioBase[CHAVE_CLASSES]
                vetorGrupos = dicionarioBase[CHAVE_GRUPOS]
                contadorDeClasses = np.bincount(vetorClasses.astype(np.int64),minlength=QUANTIDADE_AMBIENTES)

                print("Base processada salva em: " + str(CAMINHO_BASE_PROCESSADA))
                print("Formato da matriz S21: " + str(matrizS21.shape))
                print("Medições por classe: " + str(contadorDeClasses.tolist()))
                print("Quantidade de grupos físicos: " + str(np.unique(vetorGrupos).size))
                print("Tamanho do arquivo: %.2f MB" % (CAMINHO_BASE_PROCESSADA.stat().st_size / 1048576.0))
                return True

    except Exception as excecao:
        print("Erro inesperado no controlador: " + str(excecao))
        return False


if(__name__ == "__main__"):
    executar()