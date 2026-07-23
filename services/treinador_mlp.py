from __future__ import annotations

import copy
import csv
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

from model.entidades.erro_sistema import ErroSistema
from services.constantes import CHAVE_CLASSES
from services.constantes import CHAVE_INDICES_TREINO
from services.constantes import CHAVE_INDICES_VALIDACAO
from services.constantes import CHAVE_REPRESENTACAO_CARTESIANA
from services.constantes import CHAVE_REPRESENTACAO_POLAR
from services.constantes import CHAVE_REPRESENTACAO_TEMPORAL
from services.constantes import CODIGO_ERRO_DADOS_MLP_INVALIDOS
from services.constantes import CODIGO_ERRO_DESCONHECIDO
from services.constantes import CODIGO_ERRO_RESULTADO_MLP_INVALIDO
from services.constantes import CODIGO_ERRO_SALVAMENTO_MLP
from services.constantes import CODIGO_ERRO_TREINAMENTO_MLP
from services.constantes import DICIONARIO_NOMES_POR_CLASSE
from services.constantes import DIRETORIO_GRAFICOS
from services.constantes import DIRETORIO_MODELOS
from services.constantes import DIRETORIO_RESULTADOS
from services.constantes import MLP_CAMADAS_OCULTAS
from services.constantes import MLP_MAXIMO_EPOCAS
from services.constantes import MLP_MELHORA_MINIMA
from services.constantes import MLP_PACIENCIA
from services.constantes import MLP_REGULARIZACAO
from services.constantes import MLP_TAMANHO_LOTE
from services.constantes import MLP_TAXA_APRENDIZADO
from services.constantes import QUANTIDADE_AMBIENTES
from services.constantes import SEMENTE_ALEATORIA
from services.preparador_experimentos import PreparadorExperimentos


CHAVE_MODELO = "modelo"
CHAVE_NOME_REPRESENTACAO = "nome_representacao"
CHAVE_EPOCAS = "epocas"
CHAVE_PERDAS_TREINO = "perdas_treino"
CHAVE_PERDAS_VALIDACAO = "perdas_validacao"
CHAVE_ACURACIAS_VALIDACAO = "acuracias_validacao"
CHAVE_F1_VALIDACAO = "f1_validacao"
CHAVE_MELHOR_EPOCA = "melhor_epoca"
CHAVE_MELHOR_PERDA_VALIDACAO = "melhor_perda_validacao"
CHAVE_ACURACIA_FINAL = "acuracia_final"
CHAVE_F1_FINAL = "f1_final"
CHAVE_MATRIZ_CONFUSAO = "matriz_confusao"
CHAVE_TEMPO_TREINAMENTO = "tempo_treinamento"
CHAVE_QUANTIDADE_PARAMETROS = "quantidade_parametros"


class TreinadorMLP:
    __ultimoErro = None

    @staticmethod
    def __registrarErro(codigo:int,mensagem:str,origem:str,detalhe:str | None=None) -> None:
        TreinadorMLP.__ultimoErro = ErroSistema.criar(codigo,mensagem,origem,detalhe)

    @staticmethod
    def __limparUltimoErro() -> None:
        TreinadorMLP.__ultimoErro = None

    @staticmethod
    def __copiarErro(erro:ErroSistema | None,codigo:int,mensagem:str,origem:str) -> None:
        if(erro is None):
            TreinadorMLP.__registrarErro(codigo,mensagem,origem)

        else:
            TreinadorMLP.__ultimoErro = erro

    @staticmethod
    def getUltimoErro() -> ErroSistema | None:
        return TreinadorMLP.__ultimoErro

    @staticmethod
    def ultimaExecucaoDeuErro() -> bool:
        if(TreinadorMLP.__ultimoErro is None):
            return False

        else:
            return True

    @staticmethod
    def __validarIndices(vetorIndices:np.ndarray,quantidadeMedicoes:int,nomeVetor:str) -> bool:
        try:
            if(vetorIndices.ndim != 1):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_DADOS_MLP_INVALIDOS,"O vetor de índices deve possuir uma dimensão","TreinadorMLP.__validarIndices",nomeVetor)
                return False

            elif(vetorIndices.size == 0):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_DADOS_MLP_INVALIDOS,"O vetor de índices não pode ser vazio","TreinadorMLP.__validarIndices",nomeVetor)
                return False

            elif(not np.issubdtype(vetorIndices.dtype,np.integer)):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_DADOS_MLP_INVALIDOS,"O vetor de índices deve conter valores inteiros","TreinadorMLP.__validarIndices",nomeVetor)
                return False

            elif(np.any(vetorIndices < 0) or np.any(vetorIndices >= quantidadeMedicoes)):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_DADOS_MLP_INVALIDOS,"O vetor de índices contém valores fora do intervalo","TreinadorMLP.__validarIndices",nomeVetor)
                return False

            elif(np.unique(vetorIndices).size != vetorIndices.size):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_DADOS_MLP_INVALIDOS,"O vetor de índices contém valores repetidos","TreinadorMLP.__validarIndices",nomeVetor)
                return False

            else:
                return True

        except Exception as excecao:
            TreinadorMLP.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado durante a validação dos índices","TreinadorMLP.__validarIndices",str(excecao))
            return False

    @staticmethod
    def __validarDados(matrizDados:np.ndarray,vetorClasses:np.ndarray,indicesTreino:np.ndarray,indicesValidacao:np.ndarray,nomeRepresentacao:str) -> bool:
        try:
            if(matrizDados.ndim != 3):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_DADOS_MLP_INVALIDOS,"A representação deve possuir três dimensões","TreinadorMLP.__validarDados",nomeRepresentacao)
                return False

            elif(matrizDados.shape[0] == 0 or matrizDados.shape[1] == 0 or matrizDados.shape[2] == 0):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_DADOS_MLP_INVALIDOS,"A representação possui uma dimensão vazia","TreinadorMLP.__validarDados",nomeRepresentacao)
                return False

            elif(not np.all(np.isfinite(matrizDados))):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_DADOS_MLP_INVALIDOS,"A representação contém valores inválidos","TreinadorMLP.__validarDados",nomeRepresentacao)
                return False

            elif(vetorClasses.ndim != 1):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_DADOS_MLP_INVALIDOS,"O vetor de classes deve possuir uma dimensão","TreinadorMLP.__validarDados")
                return False

            elif(vetorClasses.size != matrizDados.shape[0]):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_DADOS_MLP_INVALIDOS,"A quantidade de classes não corresponde à quantidade de medições","TreinadorMLP.__validarDados")
                return False

            elif(not np.issubdtype(vetorClasses.dtype,np.integer)):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_DADOS_MLP_INVALIDOS,"O vetor de classes deve conter valores inteiros","TreinadorMLP.__validarDados")
                return False

            elif(np.any(vetorClasses < 0) or np.any(vetorClasses >= QUANTIDADE_AMBIENTES)):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_DADOS_MLP_INVALIDOS,"O vetor de classes contém valores inválidos","TreinadorMLP.__validarDados")
                return False

            elif(not isinstance(nomeRepresentacao,str)):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_DADOS_MLP_INVALIDOS,"O nome da representação é inválido","TreinadorMLP.__validarDados")
                return False

            elif(nomeRepresentacao.strip() == ""):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_DADOS_MLP_INVALIDOS,"O nome da representação não pode ser vazio","TreinadorMLP.__validarDados")
                return False

            elif(not TreinadorMLP.__validarIndices(indicesTreino,matrizDados.shape[0],"treino")):
                return False

            elif(not TreinadorMLP.__validarIndices(indicesValidacao,matrizDados.shape[0],"validação")):
                return False

            elif(np.intersect1d(indicesTreino,indicesValidacao).size != 0):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_DADOS_MLP_INVALIDOS,"Os conjuntos de treino e validação possuem medições em comum","TreinadorMLP.__validarDados")
                return False

            else:
                for classe in range(QUANTIDADE_AMBIENTES):
                    if(not np.any(vetorClasses[indicesTreino] == classe)):
                        TreinadorMLP.__registrarErro(CODIGO_ERRO_DADOS_MLP_INVALIDOS,"Uma classe não está presente no treino","TreinadorMLP.__validarDados",str(classe))
                        return False

                    elif(not np.any(vetorClasses[indicesValidacao] == classe)):
                        TreinadorMLP.__registrarErro(CODIGO_ERRO_DADOS_MLP_INVALIDOS,"Uma classe não está presente na validação","TreinadorMLP.__validarDados",str(classe))
                        return False

                return True

        except Exception as excecao:
            TreinadorMLP.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado durante a validação dos dados","TreinadorMLP.__validarDados",str(excecao))
            return False

    @staticmethod
    def __contarParametros(modelo:MLPClassifier) -> int:
        try:
            quantidadeParametros = 0

            for matrizPesos in modelo.coefs_:
                quantidadeParametros = quantidadeParametros + matrizPesos.size

            for vetorBias in modelo.intercepts_:
                quantidadeParametros = quantidadeParametros + vetorBias.size

            return int(quantidadeParametros)

        except Exception:
            return 0

    @staticmethod
    def treinar(matrizDados:np.ndarray,vetorClasses:np.ndarray,indicesTreino:np.ndarray,indicesValidacao:np.ndarray,nomeRepresentacao:str,maximoEpocas:int=MLP_MAXIMO_EPOCAS,paciencia:int=MLP_PACIENCIA,exibirProgresso:bool=True) -> dict[str,object] | None:
        TreinadorMLP.__limparUltimoErro()

        try:
            matrizDados = np.asarray(matrizDados)
            vetorClasses = np.asarray(vetorClasses)
            indicesTreino = np.asarray(indicesTreino)
            indicesValidacao = np.asarray(indicesValidacao)

            if(not TreinadorMLP.__validarDados(matrizDados,vetorClasses,indicesTreino,indicesValidacao,nomeRepresentacao)):
                return None

            elif(not isinstance(maximoEpocas,int) or isinstance(maximoEpocas,bool)):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_DADOS_MLP_INVALIDOS,"A quantidade máxima de épocas é inválida","TreinadorMLP.treinar")
                return None

            elif(maximoEpocas < 1):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_DADOS_MLP_INVALIDOS,"A quantidade máxima de épocas deve ser positiva","TreinadorMLP.treinar")
                return None

            elif(not isinstance(paciencia,int) or isinstance(paciencia,bool)):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_DADOS_MLP_INVALIDOS,"A paciência é inválida","TreinadorMLP.treinar")
                return None

            elif(paciencia < 1):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_DADOS_MLP_INVALIDOS,"A paciência deve ser positiva","TreinadorMLP.treinar")
                return None

            elif(not isinstance(exibirProgresso,bool)):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_DADOS_MLP_INVALIDOS,"O parâmetro de progresso é inválido","TreinadorMLP.treinar")
                return None

            else:
                quantidadeCaracteristicas = matrizDados.shape[1] * matrizDados.shape[2]

                matrizTreino = matrizDados[indicesTreino].reshape(indicesTreino.size,quantidadeCaracteristicas)
                matrizValidacao = matrizDados[indicesValidacao].reshape(indicesValidacao.size,quantidadeCaracteristicas)

                matrizTreino = np.ascontiguousarray(matrizTreino,dtype=np.float32)
                matrizValidacao = np.ascontiguousarray(matrizValidacao,dtype=np.float32)
                tamanhoLoteEfetivo = min(MLP_TAMANHO_LOTE,matrizTreino.shape[0])

                classesTreino = vetorClasses[indicesTreino].astype(np.int64)
                classesValidacao = vetorClasses[indicesValidacao].astype(np.int64)
                classesPossiveis = np.arange(QUANTIDADE_AMBIENTES,dtype=np.int64)

                modelo = MLPClassifier(hidden_layer_sizes=MLP_CAMADAS_OCULTAS,activation="relu",solver="adam",alpha=MLP_REGULARIZACAO,batch_size=tamanhoLoteEfetivo,learning_rate_init=MLP_TAXA_APRENDIZADO,max_iter=1,shuffle=True,random_state=SEMENTE_ALEATORIA,early_stopping=False)

                listaDeEpocas = []
                listaDePerdasTreino = []
                listaDePerdasValidacao = []
                listaDeAcuraciasValidacao = []
                listaDeF1Validacao = []

                melhorModelo = None
                melhorPerdaValidacao = np.inf
                melhorEpoca = 0
                epocasSemMelhora = 0

                instanteInicial = time.perf_counter()

                for epoca in range(1,maximoEpocas + 1):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore",category=ConvergenceWarning)

                        if(epoca == 1):
                            modelo.partial_fit(matrizTreino,classesTreino,classes=classesPossiveis)

                        else:
                            modelo.partial_fit(matrizTreino,classesTreino)

                    probabilidadesValidacao = modelo.predict_proba(matrizValidacao)
                    predicoesValidacao = modelo.predict(matrizValidacao)

                    perdaTreino = float(modelo.loss_)
                    perdaValidacao = float(log_loss(classesValidacao,probabilidadesValidacao,labels=classesPossiveis))
                    acuraciaValidacao = float(accuracy_score(classesValidacao,predicoesValidacao))
                    f1Validacao = float(f1_score(classesValidacao,predicoesValidacao,average="macro",zero_division=0.0))

                    listaDeEpocas.append(epoca)
                    listaDePerdasTreino.append(perdaTreino)
                    listaDePerdasValidacao.append(perdaValidacao)
                    listaDeAcuraciasValidacao.append(acuraciaValidacao)
                    listaDeF1Validacao.append(f1Validacao)

                    if(exibirProgresso):
                        print(nomeRepresentacao + " | época %d | perda treino %.6f | perda validação %.6f | acurácia %.4f | F1 %.4f" % (epoca,perdaTreino,perdaValidacao,acuraciaValidacao,f1Validacao))

                    if(perdaValidacao < melhorPerdaValidacao - MLP_MELHORA_MINIMA):
                        melhorPerdaValidacao = perdaValidacao
                        melhorEpoca = epoca
                        melhorModelo = copy.deepcopy(modelo)
                        epocasSemMelhora = 0

                    else:
                        epocasSemMelhora = epocasSemMelhora + 1

                    if(epocasSemMelhora >= paciencia):
                        if(exibirProgresso):
                            print(nomeRepresentacao + " | parada antecipada na época " + str(epoca))

                        break

                tempoTreinamento = time.perf_counter() - instanteInicial

                if(melhorModelo is None):
                    TreinadorMLP.__registrarErro(CODIGO_ERRO_TREINAMENTO_MLP,"Nenhum modelo válido foi obtido durante o treinamento","TreinadorMLP.treinar",nomeRepresentacao)
                    return None

                else:
                    predicoesFinais = melhorModelo.predict(matrizValidacao)
                    probabilidadesFinais = melhorModelo.predict_proba(matrizValidacao)

                    acuraciaFinal = float(accuracy_score(classesValidacao,predicoesFinais))
                    f1Final = float(f1_score(classesValidacao,predicoesFinais,average="macro",zero_division=0.0))
                    perdaFinal = float(log_loss(classesValidacao,probabilidadesFinais,labels=classesPossiveis))
                    matrizConfusao = confusion_matrix(classesValidacao,predicoesFinais,labels=classesPossiveis)

                    return {
                        CHAVE_MODELO: melhorModelo,
                        CHAVE_NOME_REPRESENTACAO: nomeRepresentacao,
                        CHAVE_EPOCAS: np.asarray(listaDeEpocas,dtype=np.int32),
                        CHAVE_PERDAS_TREINO: np.asarray(listaDePerdasTreino,dtype=np.float64),
                        CHAVE_PERDAS_VALIDACAO: np.asarray(listaDePerdasValidacao,dtype=np.float64),
                        CHAVE_ACURACIAS_VALIDACAO: np.asarray(listaDeAcuraciasValidacao,dtype=np.float64),
                        CHAVE_F1_VALIDACAO: np.asarray(listaDeF1Validacao,dtype=np.float64),
                        CHAVE_MELHOR_EPOCA: melhorEpoca,
                        CHAVE_MELHOR_PERDA_VALIDACAO: perdaFinal,
                        CHAVE_ACURACIA_FINAL: acuraciaFinal,
                        CHAVE_F1_FINAL: f1Final,
                        CHAVE_MATRIZ_CONFUSAO: matrizConfusao.astype(np.int32),
                        CHAVE_TEMPO_TREINAMENTO: float(tempoTreinamento),
                        CHAVE_QUANTIDADE_PARAMETROS: TreinadorMLP.__contarParametros(melhorModelo)
                    }

        except Exception as excecao:
            TreinadorMLP.__registrarErro(CODIGO_ERRO_TREINAMENTO_MLP,"Não foi possível treinar o modelo MLP","TreinadorMLP.treinar",str(excecao))
            return None

    @staticmethod
    def treinarRepresentacoes(dicionarioBasePreparada:dict[str,np.ndarray],exibirProgresso:bool=True) -> dict[str,dict[str,object]] | None:
        TreinadorMLP.__limparUltimoErro()

        try:
            if(not PreparadorExperimentos.validarBasePreparada(dicionarioBasePreparada)):
                TreinadorMLP.__copiarErro(PreparadorExperimentos.getUltimoErro(),CODIGO_ERRO_DADOS_MLP_INVALIDOS,"A base preparada é inválida","TreinadorMLP.treinarRepresentacoes")
                return None

            else:
                vetorClasses = np.asarray(dicionarioBasePreparada[CHAVE_CLASSES])
                indicesTreino = np.asarray(dicionarioBasePreparada[CHAVE_INDICES_TREINO])
                indicesValidacao = np.asarray(dicionarioBasePreparada[CHAVE_INDICES_VALIDACAO])

                listaDeRepresentacoes = [
                    ("cartesiana",CHAVE_REPRESENTACAO_CARTESIANA),
                    ("polar",CHAVE_REPRESENTACAO_POLAR),
                    ("temporal",CHAVE_REPRESENTACAO_TEMPORAL)
                ]

                dicionarioResultados = {}

                for item in listaDeRepresentacoes:
                    nomeRepresentacao = item[0]
                    chaveRepresentacao = item[1]
                    matrizRepresentacao = np.asarray(dicionarioBasePreparada[chaveRepresentacao])

                    print()
                    print("Treinando a representação " + nomeRepresentacao + ".")

                    resultado = TreinadorMLP.treinar(matrizRepresentacao,vetorClasses,indicesTreino,indicesValidacao,nomeRepresentacao,MLP_MAXIMO_EPOCAS,MLP_PACIENCIA,exibirProgresso)

                    if(resultado is None):
                        return None

                    else:
                        dicionarioResultados[nomeRepresentacao] = resultado

                return dicionarioResultados

        except Exception as excecao:
            TreinadorMLP.__registrarErro(CODIGO_ERRO_TREINAMENTO_MLP,"Não foi possível treinar as representações","TreinadorMLP.treinarRepresentacoes",str(excecao))
            return None

    @staticmethod
    def __validarResultado(resultado:dict[str,object]) -> bool:
        try:
            if(not isinstance(resultado,dict)):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_RESULTADO_MLP_INVALIDO,"O resultado do treinamento não é um dicionário","TreinadorMLP.__validarResultado")
                return False

            else:
                chavesObrigatorias = (
                    CHAVE_MODELO,CHAVE_NOME_REPRESENTACAO,CHAVE_EPOCAS,CHAVE_PERDAS_TREINO,
                    CHAVE_PERDAS_VALIDACAO,CHAVE_ACURACIAS_VALIDACAO,CHAVE_F1_VALIDACAO,
                    CHAVE_MELHOR_EPOCA,CHAVE_MELHOR_PERDA_VALIDACAO,CHAVE_ACURACIA_FINAL,
                    CHAVE_F1_FINAL,CHAVE_MATRIZ_CONFUSAO,CHAVE_TEMPO_TREINAMENTO,
                    CHAVE_QUANTIDADE_PARAMETROS
                )

                for chave in chavesObrigatorias:
                    if(chave not in resultado):
                        TreinadorMLP.__registrarErro(CODIGO_ERRO_RESULTADO_MLP_INVALIDO,"Uma chave obrigatória não foi encontrada no resultado","TreinadorMLP.__validarResultado",chave)
                        return False

                vetorEpocas = np.asarray(resultado[CHAVE_EPOCAS])
                vetorPerdasTreino = np.asarray(resultado[CHAVE_PERDAS_TREINO])
                vetorPerdasValidacao = np.asarray(resultado[CHAVE_PERDAS_VALIDACAO])
                vetorAcuracias = np.asarray(resultado[CHAVE_ACURACIAS_VALIDACAO])
                vetorF1 = np.asarray(resultado[CHAVE_F1_VALIDACAO])
                matrizConfusao = np.asarray(resultado[CHAVE_MATRIZ_CONFUSAO])

                if(vetorEpocas.ndim != 1 or vetorEpocas.size == 0):
                    TreinadorMLP.__registrarErro(CODIGO_ERRO_RESULTADO_MLP_INVALIDO,"O histórico de épocas é inválido","TreinadorMLP.__validarResultado")
                    return False

                elif(vetorPerdasTreino.shape != vetorEpocas.shape):
                    TreinadorMLP.__registrarErro(CODIGO_ERRO_RESULTADO_MLP_INVALIDO,"O histórico da perda de treino é inválido","TreinadorMLP.__validarResultado")
                    return False

                elif(vetorPerdasValidacao.shape != vetorEpocas.shape):
                    TreinadorMLP.__registrarErro(CODIGO_ERRO_RESULTADO_MLP_INVALIDO,"O histórico da perda de validação é inválido","TreinadorMLP.__validarResultado")
                    return False

                elif(vetorAcuracias.shape != vetorEpocas.shape or vetorF1.shape != vetorEpocas.shape):
                    TreinadorMLP.__registrarErro(CODIGO_ERRO_RESULTADO_MLP_INVALIDO,"O histórico das métricas é inválido","TreinadorMLP.__validarResultado")
                    return False

                elif(matrizConfusao.shape != (QUANTIDADE_AMBIENTES,QUANTIDADE_AMBIENTES)):
                    TreinadorMLP.__registrarErro(CODIGO_ERRO_RESULTADO_MLP_INVALIDO,"A matriz de confusão possui formato inválido","TreinadorMLP.__validarResultado")
                    return False

                elif(not np.all(np.isfinite(vetorPerdasTreino)) or not np.all(np.isfinite(vetorPerdasValidacao))):
                    TreinadorMLP.__registrarErro(CODIGO_ERRO_RESULTADO_MLP_INVALIDO,"O histórico contém valores inválidos","TreinadorMLP.__validarResultado")
                    return False

                elif(float(resultado[CHAVE_ACURACIA_FINAL]) < 0.0 or float(resultado[CHAVE_ACURACIA_FINAL]) > 1.0):
                    TreinadorMLP.__registrarErro(CODIGO_ERRO_RESULTADO_MLP_INVALIDO,"A acurácia final é inválida","TreinadorMLP.__validarResultado")
                    return False

                elif(float(resultado[CHAVE_F1_FINAL]) < 0.0 or float(resultado[CHAVE_F1_FINAL]) > 1.0):
                    TreinadorMLP.__registrarErro(CODIGO_ERRO_RESULTADO_MLP_INVALIDO,"O F1 final é inválido","TreinadorMLP.__validarResultado")
                    return False

                elif(not hasattr(resultado[CHAVE_MODELO],"predict")):
                    TreinadorMLP.__registrarErro(CODIGO_ERRO_RESULTADO_MLP_INVALIDO,"O modelo armazenado é inválido","TreinadorMLP.__validarResultado")
                    return False

                else:
                    return True

        except Exception as excecao:
            TreinadorMLP.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado durante a validação do resultado","TreinadorMLP.__validarResultado",str(excecao))
            return False

    @staticmethod
    def salvarResultados(dicionarioResultados:dict[str,dict[str,object]],diretorioResultados:Path | str=DIRETORIO_RESULTADOS,diretorioGraficos:Path | str=DIRETORIO_GRAFICOS,diretorioModelos:Path | str=DIRETORIO_MODELOS) -> list[Path] | None:
        TreinadorMLP.__limparUltimoErro()

        try:
            if(not isinstance(dicionarioResultados,dict)):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_RESULTADO_MLP_INVALIDO,"Os resultados não foram fornecidos em um dicionário","TreinadorMLP.salvarResultados")
                return None

            elif(len(dicionarioResultados) == 0):
                TreinadorMLP.__registrarErro(CODIGO_ERRO_RESULTADO_MLP_INVALIDO,"O dicionário de resultados está vazio","TreinadorMLP.salvarResultados")
                return None

            else:
                for nomeRepresentacao in dicionarioResultados:
                    if(not TreinadorMLP.__validarResultado(dicionarioResultados[nomeRepresentacao])):
                        return None

                diretorioResultados = Path(diretorioResultados)
                diretorioGraficos = Path(diretorioGraficos)
                diretorioModelos = Path(diretorioModelos)

                diretorioResultados.mkdir(parents=True,exist_ok=True)
                diretorioGraficos.mkdir(parents=True,exist_ok=True)
                diretorioModelos.mkdir(parents=True,exist_ok=True)

                listaDeArquivos = []
                listaDeNomesRepresentacoes = []
                listaDeAcuracias = []
                listaDeF1 = []

                caminhoResumo = diretorioResultados / "mlp_resumo_validacao.csv"

                with open(caminhoResumo,"w",encoding="utf-8",newline="") as arquivoCsv:
                    escritor = csv.writer(arquivoCsv)
                    escritor.writerow(["representacao","melhor_epoca","epocas_executadas","perda_validacao","acuracia_validacao","f1_macro_validacao","tempo_segundos","quantidade_parametros"])

                    for nomeRepresentacao in dicionarioResultados:
                        resultado = dicionarioResultados[nomeRepresentacao]

                        escritor.writerow([
                            nomeRepresentacao,
                            int(resultado[CHAVE_MELHOR_EPOCA]),
                            int(np.asarray(resultado[CHAVE_EPOCAS]).size),
                            "%.12f" % float(resultado[CHAVE_MELHOR_PERDA_VALIDACAO]),
                            "%.12f" % float(resultado[CHAVE_ACURACIA_FINAL]),
                            "%.12f" % float(resultado[CHAVE_F1_FINAL]),
                            "%.6f" % float(resultado[CHAVE_TEMPO_TREINAMENTO]),
                            int(resultado[CHAVE_QUANTIDADE_PARAMETROS])
                        ])

                        listaDeNomesRepresentacoes.append(nomeRepresentacao)
                        listaDeAcuracias.append(float(resultado[CHAVE_ACURACIA_FINAL]))
                        listaDeF1.append(float(resultado[CHAVE_F1_FINAL]))

                listaDeArquivos.append(caminhoResumo)

                for nomeRepresentacao in dicionarioResultados:
                    resultado = dicionarioResultados[nomeRepresentacao]

                    caminhoModelo = diretorioModelos / ("mlp_" + nomeRepresentacao + ".joblib")
                    joblib.dump(resultado[CHAVE_MODELO],caminhoModelo)
                    listaDeArquivos.append(caminhoModelo)

                    caminhoHistorico = diretorioResultados / ("mlp_historico_" + nomeRepresentacao + ".csv")

                    with open(caminhoHistorico,"w",encoding="utf-8",newline="") as arquivoCsv:
                        escritor = csv.writer(arquivoCsv)
                        escritor.writerow(["epoca","perda_treino","perda_validacao","acuracia_validacao","f1_macro_validacao"])

                        vetorEpocas = np.asarray(resultado[CHAVE_EPOCAS])
                        vetorPerdasTreino = np.asarray(resultado[CHAVE_PERDAS_TREINO])
                        vetorPerdasValidacao = np.asarray(resultado[CHAVE_PERDAS_VALIDACAO])
                        vetorAcuracias = np.asarray(resultado[CHAVE_ACURACIAS_VALIDACAO])
                        vetorF1 = np.asarray(resultado[CHAVE_F1_VALIDACAO])

                        for indiceEpoca in range(vetorEpocas.size):
                            escritor.writerow([
                                int(vetorEpocas[indiceEpoca]),
                                "%.12f" % vetorPerdasTreino[indiceEpoca],
                                "%.12f" % vetorPerdasValidacao[indiceEpoca],
                                "%.12f" % vetorAcuracias[indiceEpoca],
                                "%.12f" % vetorF1[indiceEpoca]
                            ])

                    listaDeArquivos.append(caminhoHistorico)

                    caminhoMatrizCsv = diretorioResultados / ("mlp_matriz_confusao_" + nomeRepresentacao + ".csv")
                    matrizConfusao = np.asarray(resultado[CHAVE_MATRIZ_CONFUSAO])

                    with open(caminhoMatrizCsv,"w",encoding="utf-8",newline="") as arquivoCsv:
                        escritor = csv.writer(arquivoCsv)
                        cabecalho = ["classe_real"]

                        for classePredita in range(QUANTIDADE_AMBIENTES):
                            cabecalho.append("predita_" + str(classePredita))

                        escritor.writerow(cabecalho)

                        for classeReal in range(QUANTIDADE_AMBIENTES):
                            linha = [classeReal]

                            for classePredita in range(QUANTIDADE_AMBIENTES):
                                linha.append(int(matrizConfusao[classeReal,classePredita]))

                            escritor.writerow(linha)

                    listaDeArquivos.append(caminhoMatrizCsv)

                    caminhoCurva = diretorioGraficos / ("mlp_curva_" + nomeRepresentacao + ".png")

                    plt.figure(figsize=(9,6))
                    plt.plot(resultado[CHAVE_EPOCAS],resultado[CHAVE_PERDAS_TREINO],label="Treino")
                    plt.plot(resultado[CHAVE_EPOCAS],resultado[CHAVE_PERDAS_VALIDACAO],label="Validação")
                    plt.axvline(int(resultado[CHAVE_MELHOR_EPOCA]),linestyle="--",label="Melhor época")
                    plt.xlabel("Época")
                    plt.ylabel("Entropia cruzada")
                    plt.title("Treinamento da MLP — representação " + nomeRepresentacao)
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(caminhoCurva,dpi=300,bbox_inches="tight")
                    plt.close()

                    listaDeArquivos.append(caminhoCurva)

                    caminhoMatrizGrafico = diretorioGraficos / ("mlp_matriz_confusao_" + nomeRepresentacao + ".png")

                    plt.figure(figsize=(8,7))
                    plt.imshow(matrizConfusao)
                    plt.colorbar()

                    listaDeNomesClasses = []

                    for classe in range(QUANTIDADE_AMBIENTES):
                        listaDeNomesClasses.append(DICIONARIO_NOMES_POR_CLASSE[classe])

                    plt.xticks(np.arange(QUANTIDADE_AMBIENTES),listaDeNomesClasses,rotation=30,ha="right")
                    plt.yticks(np.arange(QUANTIDADE_AMBIENTES),listaDeNomesClasses)

                    for classeReal in range(QUANTIDADE_AMBIENTES):
                        for classePredita in range(QUANTIDADE_AMBIENTES):
                            plt.text(classePredita,classeReal,str(int(matrizConfusao[classeReal,classePredita])),ha="center",va="center")

                    plt.xlabel("Classe predita")
                    plt.ylabel("Classe real")
                    plt.title("Matriz de confusão da validação — " + nomeRepresentacao)
                    plt.tight_layout()
                    plt.savefig(caminhoMatrizGrafico,dpi=300,bbox_inches="tight")
                    plt.close()

                    listaDeArquivos.append(caminhoMatrizGrafico)

                caminhoComparacao = diretorioGraficos / "mlp_comparacao_validacao.png"
                posicoes = np.arange(len(listaDeNomesRepresentacoes))
                larguraBarra = 0.35

                plt.figure(figsize=(9,6))
                plt.bar(posicoes - larguraBarra / 2.0,listaDeAcuracias,width=larguraBarra,label="Acurácia")
                plt.bar(posicoes + larguraBarra / 2.0,listaDeF1,width=larguraBarra,label="F1 macro")
                plt.xticks(posicoes,listaDeNomesRepresentacoes)
                plt.ylim(0.0,1.0)
                plt.xlabel("Representação")
                plt.ylabel("Métrica na validação")
                plt.title("Comparação das representações com a MLP")
                plt.legend()
                plt.grid(True,axis="y")
                plt.tight_layout()
                plt.savefig(caminhoComparacao,dpi=300,bbox_inches="tight")
                plt.close()

                listaDeArquivos.append(caminhoComparacao)

                return listaDeArquivos

        except Exception as excecao:
            TreinadorMLP.__registrarErro(CODIGO_ERRO_SALVAMENTO_MLP,"Não foi possível salvar os resultados da MLP","TreinadorMLP.salvarResultados",str(excecao))
            return None