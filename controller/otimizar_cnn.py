import sys

from model.entidades.erro_sistema import ErroSistema
from services.constantes import CHAVE_CLASSES
from services.constantes import CHAVE_INDICES_TREINO
from services.constantes import CHAVE_INDICES_VALIDACAO
from services.constantes import CHAVE_REPRESENTACAO_TEMPORAL
from services.constantes import OTIMIZACAO_CNN_MAXIMO_ITERACOES
from services.constantes import OTIMIZACAO_CNN_QUANTIDADE_VARIAVEIS
from services.constantes import OTIMIZACAO_CNN_TAMANHO_POPULACAO
from services.otimizador_cnn import CHAVE_CONFIGURACAO_OTIMA
from services.otimizador_cnn import CHAVE_HISTORICO_AVALIACOES
from services.otimizador_cnn import CHAVE_RESULTADO_FINAL
from services.otimizador_cnn import CHAVE_TEMPO_OTIMIZACAO
from services.otimizador_cnn import OtimizadorCNN
from services.preparador_experimentos import PreparadorExperimentos
from services.treinador_cnn import CHAVE_ACURACIA_FINAL
from services.treinador_cnn import CHAVE_F1_FINAL
from services.treinador_cnn import CHAVE_MELHOR_EPOCA
from services.treinador_cnn import CHAVE_PERDA_FINAL
from services.treinador_cnn import CHAVE_QUANTIDADE_PARAMETROS
from services.treinador_cnn import CHAVE_TEMPO_TREINAMENTO


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

        matrizTemporal = dicionarioBasePreparada[CHAVE_REPRESENTACAO_TEMPORAL]
        vetorClasses = dicionarioBasePreparada[CHAVE_CLASSES]
        indicesTreino = dicionarioBasePreparada[CHAVE_INDICES_TREINO]
        indicesValidacao = dicionarioBasePreparada[CHAVE_INDICES_VALIDACAO]

        quantidadeMaximaAvaliacoes = (OTIMIZACAO_CNN_MAXIMO_ITERACOES + 1) * OTIMIZACAO_CNN_TAMANHO_POPULACAO * OTIMIZACAO_CNN_QUANTIDADE_VARIAVEIS

        print("Iniciando a otimização da CNN 1D.")
        print("Método: evolução diferencial.")
        print("Função objetivo: perda de validação.")
        print("Quantidade máxima de avaliações: " + str(quantidadeMaximaAvaliacoes))
        print("O conjunto de teste permanecerá reservado.")
        print()

        resultadoOtimizacao = OtimizadorCNN.otimizar(matrizTemporal,vetorClasses,indicesTreino,indicesValidacao,usarCuda=True,exibirProgresso=True)

        if(resultadoOtimizacao is None):
            exibirErro(OtimizadorCNN.getUltimoErro())
            return False

        listaDeArquivos = OtimizadorCNN.salvarResultados(resultadoOtimizacao)

        if(listaDeArquivos is None):
            exibirErro(OtimizadorCNN.getUltimoErro())
            return False

        configuracaoOtima = resultadoOtimizacao[CHAVE_CONFIGURACAO_OTIMA]
        resultadoFinal = resultadoOtimizacao[CHAVE_RESULTADO_FINAL]
        historicoAvaliacoes = resultadoOtimizacao[CHAVE_HISTORICO_AVALIACOES]

        print()
        print("Otimização concluída.")
        print("Avaliações realizadas: " + str(len(historicoAvaliacoes)))
        print("Tempo total da otimização: %.2f segundos" % resultadoOtimizacao[CHAVE_TEMPO_OTIMIZACAO])

        print()
        print("Melhor configuração:")

        dicionarioConfiguracao = configuracaoOtima.paraDicionario()

        for nomeParametro in dicionarioConfiguracao:
            print(nomeParametro + ": " + str(dicionarioConfiguracao[nomeParametro]))

        print()
        print("Resultado final de validação:")
        print("Melhor época: " + str(resultadoFinal[CHAVE_MELHOR_EPOCA]))
        print("Perda: %.6f" % resultadoFinal[CHAVE_PERDA_FINAL])
        print("Acurácia: %.4f" % resultadoFinal[CHAVE_ACURACIA_FINAL])
        print("F1 macro: %.4f" % resultadoFinal[CHAVE_F1_FINAL])
        print("Tempo de treinamento final: %.2f segundos" % resultadoFinal[CHAVE_TEMPO_TREINAMENTO])
        print("Parâmetros treináveis: " + str(resultadoFinal[CHAVE_QUANTIDADE_PARAMETROS]))

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