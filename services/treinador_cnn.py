from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from model.entidades.erro_sistema import ErroSistema
from model.redes.modelo_cnn_1d import ModeloCNN1D
from services.constantes import CNN_DECAIMENTO_PESOS
from services.constantes import CNN_MAXIMO_EPOCAS
from services.constantes import CNN_MELHORA_MINIMA
from services.constantes import CNN_PACIENCIA
from services.constantes import CNN_TAMANHO_LOTE
from services.constantes import CNN_TAXA_APRENDIZADO
from services.constantes import CODIGO_ERRO_DADOS_CNN_INVALIDOS
from services.constantes import CODIGO_ERRO_DESCONHECIDO
from services.constantes import CODIGO_ERRO_RESULTADO_CNN_INVALIDO
from services.constantes import CODIGO_ERRO_SALVAMENTO_CNN
from services.constantes import CODIGO_ERRO_TREINAMENTO_CNN
from services.constantes import DICIONARIO_NOMES_POR_CLASSE
from services.constantes import DIRETORIO_GRAFICOS
from services.constantes import DIRETORIO_MODELOS
from services.constantes import DIRETORIO_RESULTADOS
from services.constantes import QUANTIDADE_AMBIENTES
from services.constantes import SEMENTE_ALEATORIA


CHAVE_MODELO = "modelo"
CHAVE_REPRESENTACAO = "representacao"
CHAVE_EPOCAS = "epocas"
CHAVE_PERDAS_TREINO = "perdas_treino"
CHAVE_PERDAS_VALIDACAO = "perdas_validacao"
CHAVE_ACURACIAS_VALIDACAO = "acuracias_validacao"
CHAVE_F1_VALIDACAO = "f1_validacao"
CHAVE_MELHOR_EPOCA = "melhor_epoca"
CHAVE_PERDA_FINAL = "perda_final"
CHAVE_ACURACIA_FINAL = "acuracia_final"
CHAVE_F1_FINAL = "f1_final"
CHAVE_MATRIZ_CONFUSAO = "matriz_confusao"
CHAVE_TEMPO_TREINAMENTO = "tempo_treinamento"
CHAVE_QUANTIDADE_PARAMETROS = "quantidade_parametros"
CHAVE_DISPOSITIVO = "dispositivo"
CHAVE_MEMORIA_MAXIMA_MB = "memoria_maxima_mb"
CHAVE_QUANTIDADE_PONTOS = "quantidade_pontos"


class TreinadorCNN:
    __ultimoErro = None

    @staticmethod
    def __registrarErro(codigo:int,mensagem:str,origem:str,detalhe:str | None=None) -> None:
        TreinadorCNN.__ultimoErro = ErroSistema.criar(codigo,mensagem,origem,detalhe)

    @staticmethod
    def __limparUltimoErro() -> None:
        TreinadorCNN.__ultimoErro = None

    @staticmethod
    def getUltimoErro() -> ErroSistema | None:
        return TreinadorCNN.__ultimoErro

    @staticmethod
    def ultimaExecucaoDeuErro() -> bool:
        if(TreinadorCNN.__ultimoErro is None):
            return False

        else:
            return True

    @staticmethod
    def __validarIndices(vetorIndices:np.ndarray,quantidadeMedicoes:int,nomeVetor:str) -> bool:
        if(vetorIndices.ndim != 1):
            TreinadorCNN.__registrarErro(CODIGO_ERRO_DADOS_CNN_INVALIDOS,"O vetor de índices deve possuir uma dimensão","TreinadorCNN.__validarIndices",nomeVetor)
            return False

        elif(vetorIndices.size == 0):
            TreinadorCNN.__registrarErro(CODIGO_ERRO_DADOS_CNN_INVALIDOS,"O vetor de índices não pode ser vazio","TreinadorCNN.__validarIndices",nomeVetor)
            return False

        elif(not np.issubdtype(vetorIndices.dtype,np.integer)):
            TreinadorCNN.__registrarErro(CODIGO_ERRO_DADOS_CNN_INVALIDOS,"O vetor de índices deve conter valores inteiros","TreinadorCNN.__validarIndices",nomeVetor)
            return False

        elif(np.any(vetorIndices < 0) or np.any(vetorIndices >= quantidadeMedicoes)):
            TreinadorCNN.__registrarErro(CODIGO_ERRO_DADOS_CNN_INVALIDOS,"O vetor de índices contém valores fora do intervalo","TreinadorCNN.__validarIndices",nomeVetor)
            return False

        elif(np.unique(vetorIndices).size != vetorIndices.size):
            TreinadorCNN.__registrarErro(CODIGO_ERRO_DADOS_CNN_INVALIDOS,"O vetor de índices contém valores repetidos","TreinadorCNN.__validarIndices",nomeVetor)
            return False

        else:
            return True

    @staticmethod
    def __validarDados(matrizDados:np.ndarray,vetorClasses:np.ndarray,indicesTreino:np.ndarray,indicesValidacao:np.ndarray) -> bool:
        try:
            if(matrizDados.ndim != 3):
                TreinadorCNN.__registrarErro(CODIGO_ERRO_DADOS_CNN_INVALIDOS,"A matriz de dados deve possuir três dimensões","TreinadorCNN.__validarDados")
                return False

            elif(matrizDados.shape[0] == 0 or matrizDados.shape[1] == 0):
                TreinadorCNN.__registrarErro(CODIGO_ERRO_DADOS_CNN_INVALIDOS,"A matriz de dados possui uma dimensão vazia","TreinadorCNN.__validarDados")
                return False

            elif(matrizDados.shape[2] != 2):
                TreinadorCNN.__registrarErro(CODIGO_ERRO_DADOS_CNN_INVALIDOS,"A matriz deve possuir dois canais de entrada","TreinadorCNN.__validarDados")
                return False

            elif(not np.all(np.isfinite(matrizDados))):
                TreinadorCNN.__registrarErro(CODIGO_ERRO_DADOS_CNN_INVALIDOS,"A matriz contém valores inválidos","TreinadorCNN.__validarDados")
                return False

            elif(vetorClasses.ndim != 1):
                TreinadorCNN.__registrarErro(CODIGO_ERRO_DADOS_CNN_INVALIDOS,"O vetor de classes deve possuir uma dimensão","TreinadorCNN.__validarDados")
                return False

            elif(vetorClasses.size != matrizDados.shape[0]):
                TreinadorCNN.__registrarErro(CODIGO_ERRO_DADOS_CNN_INVALIDOS,"A quantidade de classes não corresponde à quantidade de medições","TreinadorCNN.__validarDados")
                return False

            elif(not np.issubdtype(vetorClasses.dtype,np.integer)):
                TreinadorCNN.__registrarErro(CODIGO_ERRO_DADOS_CNN_INVALIDOS,"O vetor de classes deve conter valores inteiros","TreinadorCNN.__validarDados")
                return False

            elif(not TreinadorCNN.__validarIndices(indicesTreino,matrizDados.shape[0],"treino")):
                return False

            elif(not TreinadorCNN.__validarIndices(indicesValidacao,matrizDados.shape[0],"validação")):
                return False

            elif(np.intersect1d(indicesTreino,indicesValidacao).size != 0):
                TreinadorCNN.__registrarErro(CODIGO_ERRO_DADOS_CNN_INVALIDOS,"Os conjuntos de treino e validação possuem medições em comum","TreinadorCNN.__validarDados")
                return False

            else:
                for classe in range(QUANTIDADE_AMBIENTES):
                    if(not np.any(vetorClasses[indicesTreino] == classe)):
                        TreinadorCNN.__registrarErro(CODIGO_ERRO_DADOS_CNN_INVALIDOS,"Uma classe não está presente no treino","TreinadorCNN.__validarDados",str(classe))
                        return False

                    elif(not np.any(vetorClasses[indicesValidacao] == classe)):
                        TreinadorCNN.__registrarErro(CODIGO_ERRO_DADOS_CNN_INVALIDOS,"Uma classe não está presente na validação","TreinadorCNN.__validarDados",str(classe))
                        return False

                return True

        except Exception as excecao:
            TreinadorCNN.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado durante a validação dos dados","TreinadorCNN.__validarDados",str(excecao))
            return False

    @staticmethod
    def __configurarReprodutibilidade() -> None:
        np.random.seed(SEMENTE_ALEATORIA)
        torch.manual_seed(SEMENTE_ALEATORIA)

        if(torch.cuda.is_available()):
            torch.cuda.manual_seed_all(SEMENTE_ALEATORIA)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def __criarCarregador(matrizDados:np.ndarray,vetorClasses:np.ndarray,tamanhoLote:int,embaralhar:bool,usarMemoriaFixada:bool) -> DataLoader:
        tensorDados = torch.from_numpy(matrizDados)
        tensorClasses = torch.from_numpy(vetorClasses)
        conjuntoDados = TensorDataset(tensorDados,tensorClasses)

        gerador = None

        if(embaralhar):
            gerador = torch.Generator()
            gerador.manual_seed(SEMENTE_ALEATORIA)

        return DataLoader(conjuntoDados,batch_size=tamanhoLote,shuffle=embaralhar,num_workers=0,pin_memory=usarMemoriaFixada,drop_last=False,generator=gerador)

    @staticmethod
    def __executarEpocaTreino(modelo:ModeloCNN1D,carregador:DataLoader,funcaoPerda:nn.Module,otimizador:torch.optim.Optimizer,dispositivo:torch.device) -> float:
        modelo.train()

        perdaAcumulada = 0.0
        quantidadeMedicoes = 0

        for tensorDados,tensorClasses in carregador:
            tensorDados = tensorDados.to(dispositivo,non_blocking=True)
            tensorClasses = tensorClasses.to(dispositivo,non_blocking=True)

            otimizador.zero_grad(set_to_none=True)

            tensorSaidas = modelo(tensorDados)
            perda = funcaoPerda(tensorSaidas,tensorClasses)

            perda.backward()
            otimizador.step()

            tamanhoLoteAtual = tensorClasses.size(0)
            perdaAcumulada = perdaAcumulada + float(perda.item()) * tamanhoLoteAtual
            quantidadeMedicoes = quantidadeMedicoes + tamanhoLoteAtual

        return perdaAcumulada / quantidadeMedicoes

    @staticmethod
    def __avaliar(modelo:ModeloCNN1D,carregador:DataLoader,funcaoPerda:nn.Module,dispositivo:torch.device) -> tuple[float,np.ndarray,np.ndarray]:
        modelo.eval()

        perdaAcumulada = 0.0
        quantidadeMedicoes = 0

        listaDeClassesReais = []
        listaDeClassesPreditas = []

        with torch.inference_mode():
            for tensorDados,tensorClasses in carregador:
                tensorDados = tensorDados.to(dispositivo,non_blocking=True)
                tensorClasses = tensorClasses.to(dispositivo,non_blocking=True)

                tensorSaidas = modelo(tensorDados)
                perda = funcaoPerda(tensorSaidas,tensorClasses)
                tensorPredicoes = torch.argmax(tensorSaidas,dim=1)

                tamanhoLoteAtual = tensorClasses.size(0)
                perdaAcumulada = perdaAcumulada + float(perda.item()) * tamanhoLoteAtual
                quantidadeMedicoes = quantidadeMedicoes + tamanhoLoteAtual

                listaDeClassesReais.append(tensorClasses.detach().cpu().numpy())
                listaDeClassesPreditas.append(tensorPredicoes.detach().cpu().numpy())

        vetorClassesReais = np.concatenate(listaDeClassesReais)
        vetorClassesPreditas = np.concatenate(listaDeClassesPreditas)

        return perdaAcumulada / quantidadeMedicoes,vetorClassesReais,vetorClassesPreditas

    @staticmethod
    def treinar(matrizDados:np.ndarray,vetorClasses:np.ndarray,indicesTreino:np.ndarray,indicesValidacao:np.ndarray,nomeRepresentacao:str="temporal",maximoEpocas:int=CNN_MAXIMO_EPOCAS,paciencia:int=CNN_PACIENCIA,usarCuda:bool=True,exibirProgresso:bool=True) -> dict[str,object] | None:
        TreinadorCNN.__limparUltimoErro()

        try:
            matrizDados = np.asarray(matrizDados)
            vetorClasses = np.asarray(vetorClasses)
            indicesTreino = np.asarray(indicesTreino)
            indicesValidacao = np.asarray(indicesValidacao)

            if(not TreinadorCNN.__validarDados(matrizDados,vetorClasses,indicesTreino,indicesValidacao)):
                return None

            elif(not isinstance(maximoEpocas,int) or isinstance(maximoEpocas,bool) or maximoEpocas < 1):
                TreinadorCNN.__registrarErro(CODIGO_ERRO_DADOS_CNN_INVALIDOS,"A quantidade máxima de épocas é inválida","TreinadorCNN.treinar")
                return None

            elif(not isinstance(paciencia,int) or isinstance(paciencia,bool) or paciencia < 1):
                TreinadorCNN.__registrarErro(CODIGO_ERRO_DADOS_CNN_INVALIDOS,"A paciência é inválida","TreinadorCNN.treinar")
                return None

            elif(not isinstance(usarCuda,bool) or not isinstance(exibirProgresso,bool)):
                TreinadorCNN.__registrarErro(CODIGO_ERRO_DADOS_CNN_INVALIDOS,"Um parâmetro booleano é inválido","TreinadorCNN.treinar")
                return None

            else:
                TreinadorCNN.__configurarReprodutibilidade()

                if(usarCuda and torch.cuda.is_available()):
                    dispositivo = torch.device("cuda")

                else:
                    dispositivo = torch.device("cpu")

                matrizTreino = matrizDados[indicesTreino]
                matrizValidacao = matrizDados[indicesValidacao]

                matrizTreino = np.transpose(matrizTreino,(0,2,1))
                matrizValidacao = np.transpose(matrizValidacao,(0,2,1))

                matrizTreino = np.ascontiguousarray(matrizTreino,dtype=np.float32)
                matrizValidacao = np.ascontiguousarray(matrizValidacao,dtype=np.float32)

                classesTreino = np.ascontiguousarray(vetorClasses[indicesTreino],dtype=np.int64)
                classesValidacao = np.ascontiguousarray(vetorClasses[indicesValidacao],dtype=np.int64)

                tamanhoLoteTreino = min(CNN_TAMANHO_LOTE,indicesTreino.size)
                tamanhoLoteValidacao = min(CNN_TAMANHO_LOTE,indicesValidacao.size)
                usarMemoriaFixada = dispositivo.type == "cuda"

                carregadorTreino = TreinadorCNN.__criarCarregador(matrizTreino,classesTreino,tamanhoLoteTreino,True,usarMemoriaFixada)
                carregadorValidacao = TreinadorCNN.__criarCarregador(matrizValidacao,classesValidacao,tamanhoLoteValidacao,False,usarMemoriaFixada)

                modelo = ModeloCNN1D()
                modelo = modelo.to(dispositivo)

                funcaoPerda = nn.CrossEntropyLoss()
                otimizador = torch.optim.Adam(modelo.parameters(),lr=CNN_TAXA_APRENDIZADO,weight_decay=CNN_DECAIMENTO_PESOS)

                listaDeEpocas = []
                listaDePerdasTreino = []
                listaDePerdasValidacao = []
                listaDeAcuraciasValidacao = []
                listaDeF1Validacao = []

                melhorEstado = None
                melhorEpoca = 0
                melhorPerdaValidacao = np.inf
                epocasSemMelhora = 0

                if(dispositivo.type == "cuda"):
                    torch.cuda.reset_peak_memory_stats()

                instanteInicial = time.perf_counter()

                for epoca in range(1,maximoEpocas + 1):
                    perdaTreino = TreinadorCNN.__executarEpocaTreino(modelo,carregadorTreino,funcaoPerda,otimizador,dispositivo)
                    perdaValidacao,classesReais,classesPreditas = TreinadorCNN.__avaliar(modelo,carregadorValidacao,funcaoPerda,dispositivo)

                    acuraciaValidacao = float(accuracy_score(classesReais,classesPreditas))
                    f1Validacao = float(f1_score(classesReais,classesPreditas,average="macro",zero_division=0.0))

                    listaDeEpocas.append(epoca)
                    listaDePerdasTreino.append(perdaTreino)
                    listaDePerdasValidacao.append(perdaValidacao)
                    listaDeAcuraciasValidacao.append(acuraciaValidacao)
                    listaDeF1Validacao.append(f1Validacao)

                    if(exibirProgresso):
                        print(nomeRepresentacao + " | época %d | perda treino %.6f | perda validação %.6f | acurácia %.4f | F1 %.4f" % (epoca,perdaTreino,perdaValidacao,acuraciaValidacao,f1Validacao))

                    if(perdaValidacao < melhorPerdaValidacao - CNN_MELHORA_MINIMA):
                        melhorEstado = {}

                        for chave,valor in modelo.state_dict().items():
                            melhorEstado[chave] = valor.detach().cpu().clone()

                        melhorPerdaValidacao = perdaValidacao
                        melhorEpoca = epoca
                        epocasSemMelhora = 0

                    else:
                        epocasSemMelhora = epocasSemMelhora + 1

                    if(epocasSemMelhora >= paciencia):
                        if(exibirProgresso):
                            print(nomeRepresentacao + " | parada antecipada na época " + str(epoca))

                        break

                tempoTreinamento = time.perf_counter() - instanteInicial

                if(melhorEstado is None):
                    TreinadorCNN.__registrarErro(CODIGO_ERRO_TREINAMENTO_CNN,"Nenhum modelo válido foi obtido durante o treinamento","TreinadorCNN.treinar")
                    return None

                else:
                    modelo.load_state_dict(melhorEstado)

                    perdaFinal,classesReais,classesPreditas = TreinadorCNN.__avaliar(modelo,carregadorValidacao,funcaoPerda,dispositivo)

                    acuraciaFinal = float(accuracy_score(classesReais,classesPreditas))
                    f1Final = float(f1_score(classesReais,classesPreditas,average="macro",zero_division=0.0))
                    matrizConfusao = confusion_matrix(classesReais,classesPreditas,labels=np.arange(QUANTIDADE_AMBIENTES))

                    memoriaMaximaMb = 0.0

                    if(dispositivo.type == "cuda"):
                        memoriaMaximaMb = torch.cuda.max_memory_allocated() / 1048576.0
                        nomeDispositivo = torch.cuda.get_device_name(0)

                    else:
                        nomeDispositivo = "CPU"

                    quantidadeParametros = modelo.contarParametrosTreinaveis()

                    modelo = modelo.to("cpu")

                    if(dispositivo.type == "cuda"):
                        torch.cuda.empty_cache()

                    return {
                        CHAVE_MODELO: modelo,
                        CHAVE_REPRESENTACAO: nomeRepresentacao,
                        CHAVE_EPOCAS: np.asarray(listaDeEpocas,dtype=np.int32),
                        CHAVE_PERDAS_TREINO: np.asarray(listaDePerdasTreino,dtype=np.float64),
                        CHAVE_PERDAS_VALIDACAO: np.asarray(listaDePerdasValidacao,dtype=np.float64),
                        CHAVE_ACURACIAS_VALIDACAO: np.asarray(listaDeAcuraciasValidacao,dtype=np.float64),
                        CHAVE_F1_VALIDACAO: np.asarray(listaDeF1Validacao,dtype=np.float64),
                        CHAVE_MELHOR_EPOCA: melhorEpoca,
                        CHAVE_PERDA_FINAL: float(perdaFinal),
                        CHAVE_ACURACIA_FINAL: acuraciaFinal,
                        CHAVE_F1_FINAL: f1Final,
                        CHAVE_MATRIZ_CONFUSAO: matrizConfusao.astype(np.int32),
                        CHAVE_TEMPO_TREINAMENTO: float(tempoTreinamento),
                        CHAVE_QUANTIDADE_PARAMETROS: quantidadeParametros,
                        CHAVE_DISPOSITIVO: nomeDispositivo,
                        CHAVE_MEMORIA_MAXIMA_MB: float(memoriaMaximaMb),
                        CHAVE_QUANTIDADE_PONTOS: matrizDados.shape[1]
                    }

        except Exception as excecao:
            TreinadorCNN.__registrarErro(CODIGO_ERRO_TREINAMENTO_CNN,"Não foi possível treinar a CNN","TreinadorCNN.treinar",str(excecao))
            return None

    @staticmethod
    def __validarResultado(resultado:dict[str,object]) -> bool:
        try:
            if(not isinstance(resultado,dict)):
                TreinadorCNN.__registrarErro(CODIGO_ERRO_RESULTADO_CNN_INVALIDO,"O resultado não é um dicionário","TreinadorCNN.__validarResultado")
                return False

            else:
                chavesObrigatorias = (
                    CHAVE_MODELO,CHAVE_REPRESENTACAO,CHAVE_EPOCAS,CHAVE_PERDAS_TREINO,
                    CHAVE_PERDAS_VALIDACAO,CHAVE_ACURACIAS_VALIDACAO,CHAVE_F1_VALIDACAO,
                    CHAVE_MELHOR_EPOCA,CHAVE_PERDA_FINAL,CHAVE_ACURACIA_FINAL,
                    CHAVE_F1_FINAL,CHAVE_MATRIZ_CONFUSAO,CHAVE_TEMPO_TREINAMENTO,
                    CHAVE_QUANTIDADE_PARAMETROS,CHAVE_DISPOSITIVO,CHAVE_MEMORIA_MAXIMA_MB,
                    CHAVE_QUANTIDADE_PONTOS
                )

                for chave in chavesObrigatorias:
                    if(chave not in resultado):
                        TreinadorCNN.__registrarErro(CODIGO_ERRO_RESULTADO_CNN_INVALIDO,"Uma chave obrigatória não foi encontrada","TreinadorCNN.__validarResultado",chave)
                        return False

                vetorEpocas = np.asarray(resultado[CHAVE_EPOCAS])
                vetorPerdasTreino = np.asarray(resultado[CHAVE_PERDAS_TREINO])
                vetorPerdasValidacao = np.asarray(resultado[CHAVE_PERDAS_VALIDACAO])
                vetorAcuracias = np.asarray(resultado[CHAVE_ACURACIAS_VALIDACAO])
                vetorF1 = np.asarray(resultado[CHAVE_F1_VALIDACAO])
                matrizConfusao = np.asarray(resultado[CHAVE_MATRIZ_CONFUSAO])

                if(vetorEpocas.ndim != 1 or vetorEpocas.size == 0):
                    TreinadorCNN.__registrarErro(CODIGO_ERRO_RESULTADO_CNN_INVALIDO,"O histórico de épocas é inválido","TreinadorCNN.__validarResultado")
                    return False

                elif(vetorPerdasTreino.shape != vetorEpocas.shape or vetorPerdasValidacao.shape != vetorEpocas.shape):
                    TreinadorCNN.__registrarErro(CODIGO_ERRO_RESULTADO_CNN_INVALIDO,"O histórico de perdas é inválido","TreinadorCNN.__validarResultado")
                    return False

                elif(vetorAcuracias.shape != vetorEpocas.shape or vetorF1.shape != vetorEpocas.shape):
                    TreinadorCNN.__registrarErro(CODIGO_ERRO_RESULTADO_CNN_INVALIDO,"O histórico de métricas é inválido","TreinadorCNN.__validarResultado")
                    return False

                elif(matrizConfusao.shape != (QUANTIDADE_AMBIENTES,QUANTIDADE_AMBIENTES)):
                    TreinadorCNN.__registrarErro(CODIGO_ERRO_RESULTADO_CNN_INVALIDO,"A matriz de confusão é inválida","TreinadorCNN.__validarResultado")
                    return False

                elif(not isinstance(resultado[CHAVE_MODELO],ModeloCNN1D)):
                    TreinadorCNN.__registrarErro(CODIGO_ERRO_RESULTADO_CNN_INVALIDO,"O modelo armazenado é inválido","TreinadorCNN.__validarResultado")
                    return False

                else:
                    return True

        except Exception as excecao:
            TreinadorCNN.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado durante a validação do resultado","TreinadorCNN.__validarResultado",str(excecao))
            return False

    @staticmethod
    def salvarResultados(resultado:dict[str,object],diretorioResultados:Path | str=DIRETORIO_RESULTADOS,diretorioGraficos:Path | str=DIRETORIO_GRAFICOS,diretorioModelos:Path | str=DIRETORIO_MODELOS) -> list[Path] | None:
        TreinadorCNN.__limparUltimoErro()

        try:
            if(not TreinadorCNN.__validarResultado(resultado)):
                return None

            else:
                diretorioResultados = Path(diretorioResultados)
                diretorioGraficos = Path(diretorioGraficos)
                diretorioModelos = Path(diretorioModelos)

                diretorioResultados.mkdir(parents=True,exist_ok=True)
                diretorioGraficos.mkdir(parents=True,exist_ok=True)
                diretorioModelos.mkdir(parents=True,exist_ok=True)

                listaDeArquivos = []

                caminhoResumo = diretorioResultados / "cnn_resumo_validacao.csv"

                with open(caminhoResumo,"w",encoding="utf-8",newline="") as arquivoCsv:
                    escritor = csv.writer(arquivoCsv)
                    escritor.writerow(["representacao","melhor_epoca","epocas_executadas","perda_validacao","acuracia_validacao","f1_macro_validacao","tempo_segundos","quantidade_parametros","dispositivo","memoria_maxima_mb"])
                    escritor.writerow([
                        resultado[CHAVE_REPRESENTACAO],
                        int(resultado[CHAVE_MELHOR_EPOCA]),
                        int(np.asarray(resultado[CHAVE_EPOCAS]).size),
                        "%.12f" % float(resultado[CHAVE_PERDA_FINAL]),
                        "%.12f" % float(resultado[CHAVE_ACURACIA_FINAL]),
                        "%.12f" % float(resultado[CHAVE_F1_FINAL]),
                        "%.6f" % float(resultado[CHAVE_TEMPO_TREINAMENTO]),
                        int(resultado[CHAVE_QUANTIDADE_PARAMETROS]),
                        resultado[CHAVE_DISPOSITIVO],
                        "%.6f" % float(resultado[CHAVE_MEMORIA_MAXIMA_MB])
                    ])

                listaDeArquivos.append(caminhoResumo)

                caminhoHistorico = diretorioResultados / "cnn_historico_temporal.csv"

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

                matrizConfusao = np.asarray(resultado[CHAVE_MATRIZ_CONFUSAO])
                caminhoMatrizCsv = diretorioResultados / "cnn_matriz_confusao_temporal.csv"

                with open(caminhoMatrizCsv,"w",encoding="utf-8",newline="") as arquivoCsv:
                    escritor = csv.writer(arquivoCsv)
                    escritor.writerow(["classe_real","predita_0","predita_1","predita_2","predita_3"])

                    for classeReal in range(QUANTIDADE_AMBIENTES):
                        linha = [classeReal]

                        for classePredita in range(QUANTIDADE_AMBIENTES):
                            linha.append(int(matrizConfusao[classeReal,classePredita]))

                        escritor.writerow(linha)

                listaDeArquivos.append(caminhoMatrizCsv)

                caminhoModelo = diretorioModelos / "cnn_temporal.pt"

                pacoteModelo = {
                    "estado_modelo": resultado[CHAVE_MODELO].state_dict(),
                    "representacao": resultado[CHAVE_REPRESENTACAO],
                    "quantidade_classes": QUANTIDADE_AMBIENTES,
                    "quantidade_pontos": int(resultado[CHAVE_QUANTIDADE_PONTOS])
                }

                torch.save(pacoteModelo,caminhoModelo)
                listaDeArquivos.append(caminhoModelo)

                caminhoCurva = diretorioGraficos / "cnn_curva_temporal.png"

                plt.figure(figsize=(9,6))
                plt.plot(vetorEpocas,vetorPerdasTreino,label="Treino")
                plt.plot(vetorEpocas,vetorPerdasValidacao,label="Validação")
                plt.axvline(int(resultado[CHAVE_MELHOR_EPOCA]),linestyle="--",label="Melhor época")
                plt.xlabel("Época")
                plt.ylabel("Entropia cruzada")
                plt.title("Treinamento da CNN 1D — representação temporal")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(caminhoCurva,dpi=300,bbox_inches="tight")
                plt.close()

                listaDeArquivos.append(caminhoCurva)

                caminhoMatrizGrafico = diretorioGraficos / "cnn_matriz_confusao_temporal.png"
                listaDeNomesClasses = []

                for classe in range(QUANTIDADE_AMBIENTES):
                    listaDeNomesClasses.append(DICIONARIO_NOMES_POR_CLASSE[classe])

                plt.figure(figsize=(8,7))
                plt.imshow(matrizConfusao)
                plt.colorbar()
                plt.xticks(np.arange(QUANTIDADE_AMBIENTES),listaDeNomesClasses,rotation=30,ha="right")
                plt.yticks(np.arange(QUANTIDADE_AMBIENTES),listaDeNomesClasses)

                for classeReal in range(QUANTIDADE_AMBIENTES):
                    for classePredita in range(QUANTIDADE_AMBIENTES):
                        plt.text(classePredita,classeReal,str(int(matrizConfusao[classeReal,classePredita])),ha="center",va="center")

                plt.xlabel("Classe predita")
                plt.ylabel("Classe real")
                plt.title("Matriz de confusão da validação — CNN 1D")
                plt.tight_layout()
                plt.savefig(caminhoMatrizGrafico,dpi=300,bbox_inches="tight")
                plt.close()

                listaDeArquivos.append(caminhoMatrizGrafico)

                caminhoResumoMlp = diretorioResultados / "mlp_resumo_validacao.csv"

                if(not caminhoResumoMlp.exists()):
                    TreinadorCNN.__registrarErro(CODIGO_ERRO_SALVAMENTO_CNN,"O resumo da MLP não foi encontrado","TreinadorCNN.salvarResultados",str(caminhoResumoMlp))
                    return None

                linhaMlpTemporal = None

                with open(caminhoResumoMlp,"r",encoding="utf-8",newline="") as arquivoCsv:
                    leitor = csv.DictReader(arquivoCsv)

                    for linha in leitor:
                        if(linha["representacao"] == "temporal"):
                            linhaMlpTemporal = linha
                            break

                if(linhaMlpTemporal is None):
                    TreinadorCNN.__registrarErro(CODIGO_ERRO_SALVAMENTO_CNN,"O resultado temporal da MLP não foi encontrado","TreinadorCNN.salvarResultados")
                    return None

                caminhoComparacaoCsv = diretorioResultados / "comparacao_mlp_cnn_validacao.csv"

                with open(caminhoComparacaoCsv,"w",encoding="utf-8",newline="") as arquivoCsv:
                    escritor = csv.writer(arquivoCsv)
                    escritor.writerow(["modelo","representacao","perda_validacao","acuracia_validacao","f1_macro_validacao","tempo_segundos","quantidade_parametros"])
                    escritor.writerow([
                        "MLP",
                        "temporal",
                        linhaMlpTemporal["perda_validacao"],
                        linhaMlpTemporal["acuracia_validacao"],
                        linhaMlpTemporal["f1_macro_validacao"],
                        linhaMlpTemporal["tempo_segundos"],
                        linhaMlpTemporal["quantidade_parametros"]
                    ])
                    escritor.writerow([
                        "CNN 1D",
                        resultado[CHAVE_REPRESENTACAO],
                        "%.12f" % float(resultado[CHAVE_PERDA_FINAL]),
                        "%.12f" % float(resultado[CHAVE_ACURACIA_FINAL]),
                        "%.12f" % float(resultado[CHAVE_F1_FINAL]),
                        "%.6f" % float(resultado[CHAVE_TEMPO_TREINAMENTO]),
                        int(resultado[CHAVE_QUANTIDADE_PARAMETROS])
                    ])

                listaDeArquivos.append(caminhoComparacaoCsv)

                caminhoComparacaoGrafico = diretorioGraficos / "comparacao_mlp_cnn_validacao.png"

                nomesModelos = ["MLP","CNN 1D"]
                acuracias = [float(linhaMlpTemporal["acuracia_validacao"]),float(resultado[CHAVE_ACURACIA_FINAL])]
                valoresF1 = [float(linhaMlpTemporal["f1_macro_validacao"]),float(resultado[CHAVE_F1_FINAL])]

                posicoes = np.arange(2)
                larguraBarra = 0.35

                plt.figure(figsize=(8,6))
                plt.bar(posicoes - larguraBarra / 2.0,acuracias,width=larguraBarra,label="Acurácia")
                plt.bar(posicoes + larguraBarra / 2.0,valoresF1,width=larguraBarra,label="F1 macro")
                plt.xticks(posicoes,nomesModelos)
                plt.ylim(0.0,1.0)
                plt.xlabel("Modelo")
                plt.ylabel("Métrica de validação")
                plt.title("Comparação entre MLP e CNN 1D")
                plt.legend()
                plt.grid(True,axis="y")
                plt.tight_layout()
                plt.savefig(caminhoComparacaoGrafico,dpi=300,bbox_inches="tight")
                plt.close()

                listaDeArquivos.append(caminhoComparacaoGrafico)

                return listaDeArquivos

        except Exception as excecao:
            TreinadorCNN.__registrarErro(CODIGO_ERRO_SALVAMENTO_CNN,"Não foi possível salvar os resultados da CNN","TreinadorCNN.salvarResultados",str(excecao))
            return None