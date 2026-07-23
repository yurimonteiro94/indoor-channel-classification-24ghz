import sys

from model.entidades.erro_sistema import ErroSistema
from services.constantes import CHAVE_CLASSES
from services.constantes import CHAVE_INDICES_TREINO
from services.constantes import CHAVE_INDICES_VALIDACAO
from services.constantes import CHAVE_REPRESENTACAO_TEMPORAL
from services.preparador_experimentos import PreparadorExperimentos
from services.treinador_cnn import CHAVE_ACURACIA_FINAL
from services.treinador_cnn import CHAVE_DISPOSITIVO
from services.treinador_cnn import CHAVE_F1_FINAL
from services.treinador_cnn import CHAVE_MELHOR_EPOCA
from services.treinador_cnn import CHAVE_MEMORIA_MAXIMA_MB
from services.treinador_cnn import CHAVE_PERDA_FINAL
from services.treinador_cnn import CHAVE_QUANTIDADE_PARAMETROS
from services.treinador_cnn import CHAVE_TEMPO_TREINAMENTO
from services.treinador_cnn import TreinadorCNN


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
        print("Carregando a base preparada.")

        dicionarioBasePreparada = PreparadorExperimentos.carregarBasePreparada()

        if(dicionarioBasePreparada is None):
            exibirErro(PreparadorExperimentos.getUltimoErro())
            return False

        else:
            matrizTemporal = dicionarioBasePreparada[CHAVE_REPRESENTACAO_TEMPORAL]
            vetorClasses = dicionarioBasePreparada[CHAVE_CLASSES]
            indicesTreino = dicionarioBasePreparada[CHAVE_INDICES_TREINO]
            indicesValidacao = dicionarioBasePreparada[CHAVE_INDICES_VALIDACAO]

            print("Iniciando o treinamento da CNN 1D.")
            print("Representação utilizada: temporal.")
            print("O conjunto de teste permanecerá reservado.")

            resultado = TreinadorCNN.treinar(matrizTemporal,vetorClasses,indicesTreino,indicesValidacao,"temporal",usarCuda=True,exibirProgresso=True)

            if(resultado is None):
                exibirErro(TreinadorCNN.getUltimoErro())
                return False

            else:
                listaDeArquivos = TreinadorCNN.salvarResultados(resultado)

                if(listaDeArquivos is None):
                    exibirErro(TreinadorCNN.getUltimoErro())
                    return False

                else:
                    print()
                    print("Resultado de validação da CNN 1D:")
                    print("Melhor época: " + str(resultado[CHAVE_MELHOR_EPOCA]))
                    print("Perda de validação: %.6f" % resultado[CHAVE_PERDA_FINAL])
                    print("Acurácia de validação: %.4f" % resultado[CHAVE_ACURACIA_FINAL])
                    print("F1 macro de validação: %.4f" % resultado[CHAVE_F1_FINAL])
                    print("Tempo de treinamento: %.2f segundos" % resultado[CHAVE_TEMPO_TREINAMENTO])
                    print("Parâmetros treináveis: " + str(resultado[CHAVE_QUANTIDADE_PARAMETROS]))
                    print("Dispositivo: " + str(resultado[CHAVE_DISPOSITIVO]))
                    print("Memória máxima da GPU: %.2f MB" % resultado[CHAVE_MEMORIA_MAXIMA_MB])

                    print()
                    print("Arquivos gerados: " + str(len(listaDeArquivos)))

                    for caminhoArquivo in listaDeArquivos:
                        print(str(caminhoArquivo))

                    print()
                    print("O conjunto de teste não foi utilizado.")

                    return True

    except Exception as excecao:
        print("Erro inesperado no controlador: " + str(excecao))
        return False


if(__name__ == "__main__"):
    if(executar()):
        sys.exit(0)

    else:
        sys.exit(1)