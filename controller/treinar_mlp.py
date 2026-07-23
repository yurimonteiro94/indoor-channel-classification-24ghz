import sys

from model.entidades.erro_sistema import ErroSistema
from services.preparador_experimentos import PreparadorExperimentos
from services.treinador_mlp import CHAVE_ACURACIA_FINAL
from services.treinador_mlp import CHAVE_F1_FINAL
from services.treinador_mlp import CHAVE_MELHOR_EPOCA
from services.treinador_mlp import CHAVE_MELHOR_PERDA_VALIDACAO
from services.treinador_mlp import CHAVE_QUANTIDADE_PARAMETROS
from services.treinador_mlp import CHAVE_TEMPO_TREINAMENTO
from services.treinador_mlp import TreinadorMLP


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
            print("Iniciando o treinamento das redes MLP.")
            print("O conjunto de teste permanecerá reservado.")

            dicionarioResultados = TreinadorMLP.treinarRepresentacoes(dicionarioBasePreparada,True)

            if(dicionarioResultados is None):
                exibirErro(TreinadorMLP.getUltimoErro())
                return False

            else:
                listaDeArquivos = TreinadorMLP.salvarResultados(dicionarioResultados)

                if(listaDeArquivos is None):
                    exibirErro(TreinadorMLP.getUltimoErro())
                    return False

                else:
                    print()
                    print("Resultados de validação:")

                    for nomeRepresentacao in dicionarioResultados:
                        resultado = dicionarioResultados[nomeRepresentacao]

                        print()
                        print("Representação: " + nomeRepresentacao)
                        print("Melhor época: " + str(resultado[CHAVE_MELHOR_EPOCA]))
                        print("Perda de validação: %.6f" % resultado[CHAVE_MELHOR_PERDA_VALIDACAO])
                        print("Acurácia de validação: %.4f" % resultado[CHAVE_ACURACIA_FINAL])
                        print("F1 macro de validação: %.4f" % resultado[CHAVE_F1_FINAL])
                        print("Tempo de treinamento: %.2f segundos" % resultado[CHAVE_TEMPO_TREINAMENTO])
                        print("Parâmetros treináveis: " + str(resultado[CHAVE_QUANTIDADE_PARAMETROS]))

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