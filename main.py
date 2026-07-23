import subprocess
import sys


COMANDO_AJUDA = "ajuda"
COMANDO_BAIXAR_BASE = "baixar-base"
COMANDO_PROCESSAR_BASE = "processar-base"
COMANDO_PREPARAR_EXPERIMENTOS = "preparar-experimentos"
COMANDO_GERAR_ANALISE = "gerar-analise"
COMANDO_TREINAR_MLP = "treinar-mlp"
COMANDO_EXECUTAR_PIPELINE = "executar-pipeline"
COMANDO_TESTAR = "testar"


def exibirAjuda() -> None:
    print("Classificação de ambientes internos por canal sem fio em 2,4 GHz")
    print()
    print("Uso:")
    print("python main.py <comando>")
    print()
    print("Comandos disponíveis:")
    print("ajuda                  Exibe esta mensagem.")
    print("baixar-base            Baixa ou reutiliza a base original da UCI.")
    print("processar-base         Processa os arquivos CSV da base original.")
    print("preparar-experimentos  Cria as divisões e representações normalizadas.")
    print("gerar-analise          Gera a análise exploratória, tabelas e gráficos.")
    print("treinar-mlp            Treina e compara as redes MLP.")
    print("executar-pipeline      Executa todas as etapas em sequência.")
    print("testar                  Executa todos os testes automatizados.")


def baixarBase() -> bool:
    try:
        from ferramentas.baixar_base import executar

        return executar()

    except Exception as excecao:
        print("Não foi possível executar o download da base: " + str(excecao))
        return False


def processarBase() -> bool:
    try:
        from controller.processar_base import executar

        return executar()

    except Exception as excecao:
        print("Não foi possível processar a base: " + str(excecao))
        return False


def prepararExperimentos() -> bool:
    try:
        from controller.preparar_experimentos import executar

        return executar()

    except Exception as excecao:
        print("Não foi possível preparar os experimentos: " + str(excecao))
        return False


def gerarAnaliseExploratoria() -> bool:
    try:
        from controller.gerar_analise_exploratoria import executar

        return executar()

    except Exception as excecao:
        print("Não foi possível gerar a análise exploratória: " + str(excecao))
        return False


def treinarMLP() -> bool:
    try:
        from controller.treinar_mlp import executar

        return executar()

    except Exception as excecao:
        print("Não foi possível treinar as redes MLP: " + str(excecao))
        return False


def executarTestes() -> bool:
    try:
        listaDeComandos = [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "test",
            "-p",
            "test_*.py",
            "-v"
        ]

        resultado = subprocess.run(listaDeComandos,check=False)

        if(resultado.returncode == 0):
            return True

        else:
            return False

    except Exception as excecao:
        print("Não foi possível executar os testes: " + str(excecao))
        return False


def executarPipeline() -> bool:
    print("Etapa 1 de 5: download da base.")

    if(not baixarBase()):
        return False

    print()
    print("Etapa 2 de 5: processamento da base.")

    if(not processarBase()):
        return False

    print()
    print("Etapa 3 de 5: preparação dos experimentos.")

    if(not prepararExperimentos()):
        return False

    print()
    print("Etapa 4 de 5: análise exploratória.")

    if(not gerarAnaliseExploratoria()):
        return False

    print()
    print("Etapa 5 de 5: treinamento das redes MLP.")

    if(not treinarMLP()):
        return False

    print()
    print("Pipeline concluído com sucesso.")

    return True


def executarComando(comando:str) -> bool:
    if(comando == COMANDO_AJUDA):
        exibirAjuda()
        return True

    elif(comando == COMANDO_BAIXAR_BASE):
        return baixarBase()

    elif(comando == COMANDO_PROCESSAR_BASE):
        return processarBase()

    elif(comando == COMANDO_PREPARAR_EXPERIMENTOS):
        return prepararExperimentos()

    elif(comando == COMANDO_GERAR_ANALISE):
        return gerarAnaliseExploratoria()

    elif(comando == COMANDO_TREINAR_MLP):
        return treinarMLP()

    elif(comando == COMANDO_EXECUTAR_PIPELINE):
        return executarPipeline()

    elif(comando == COMANDO_TESTAR):
        return executarTestes()

    else:
        print("Comando desconhecido: " + comando)
        print()
        exibirAjuda()
        return False


def main() -> int:
    try:
        if(len(sys.argv) == 1):
            exibirAjuda()
            return 0

        elif(len(sys.argv) != 2):
            print("A quantidade de argumentos é inválida.")
            print()
            exibirAjuda()
            return 1

        else:
            comando = sys.argv[1].strip().lower()

            if(executarComando(comando)):
                return 0

            else:
                return 1

    except Exception as excecao:
        print("Erro inesperado na aplicação: " + str(excecao))
        return 1


if(__name__ == "__main__"):
    sys.exit(main())