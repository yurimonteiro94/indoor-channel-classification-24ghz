import sys

from model.entidades.erro_sistema import ErroSistema
from services.analisador_exploratorio import AnalisadorExploratorio


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
        print("Iniciando a análise exploratória.")

        listaDeArquivos = AnalisadorExploratorio.executarAnalise()

        if(listaDeArquivos is None):
            exibirErro(AnalisadorExploratorio.getUltimoErro())
            return False

        else:
            print("Análise exploratória concluída.")
            print("Arquivos gerados: " + str(len(listaDeArquivos)))

            for caminhoArquivo in listaDeArquivos:
                print(str(caminhoArquivo))

            return True

    except Exception as excecao:
        print("Erro inesperado no controlador: " + str(excecao))
        return False


if(__name__ == "__main__"):
    if(executar()):
        sys.exit(0)

    else:
        sys.exit(1)