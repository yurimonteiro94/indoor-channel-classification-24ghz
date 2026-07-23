from __future__ import annotations

import csv
import time
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import differential_evolution

from model.entidades.erro_sistema import ErroSistema
from model.redes.modelo_cnn_1d import ConfiguracaoCNN
from services.constantes import CNN_MAXIMO_EPOCAS
from services.constantes import CNN_PACIENCIA
from services.constantes import CODIGO_ERRO_DESCONHECIDO
from services.constantes import CODIGO_ERRO_OTIMIZACAO_CNN
from services.constantes import CODIGO_ERRO_RESULTADO_OTIMIZACAO_INVALIDO
from services.constantes import CODIGO_ERRO_SALVAMENTO_OTIMIZACAO
from services.constantes import CODIGO_ERRO_VETOR_OTIMIZACAO_INVALIDO
from services.constantes import DICIONARIO_NOMES_POR_CLASSE
from services.constantes import DIRETORIO_GRAFICOS
from services.constantes import DIRETORIO_MODELOS
from services.constantes import DIRETORIO_RESULTADOS
from services.constantes import OTIMIZACAO_CNN_MAXIMO_EPOCAS_AVALIACAO
from services.constantes import OTIMIZACAO_CNN_MAXIMO_ITERACOES
from services.constantes import OTIMIZACAO_CNN_PACIENCIA_AVALIACAO
from services.constantes import OTIMIZACAO_CNN_PENALIDADE
from services.constantes import OTIMIZACAO_CNN_TAMANHO_POPULACAO
from services.constantes import QUANTIDADE_AMBIENTES
from services.constantes import SEMENTE_ALEATORIA
from services.treinador_cnn import CHAVE_ACURACIA_FINAL
from services.treinador_cnn import CHAVE_CONFIGURACAO
from services.treinador_cnn import CHAVE_DISPOSITIVO
from services.treinador_cnn import CHAVE_EPOCAS
from services.treinador_cnn import CHAVE_F1_FINAL
from services.treinador_cnn import CHAVE_MATRIZ_CONFUSAO
from services.treinador_cnn import CHAVE_MELHOR_EPOCA
from services.treinador_cnn import CHAVE_MEMORIA_MAXIMA_MB
from services.treinador_cnn import CHAVE_MODELO
from services.treinador_cnn import CHAVE_PERDA_FINAL
from services.treinador_cnn import CHAVE_PERDAS_TREINO
from services.treinador_cnn import CHAVE_PERDAS_VALIDACAO
from services.treinador_cnn import CHAVE_QUANTIDADE_PARAMETROS
from services.treinador_cnn import CHAVE_QUANTIDADE_PONTOS
from services.treinador_cnn import CHAVE_TEMPO_TREINAMENTO
from services.treinador_cnn import TreinadorCNN


CHAVE_CONFIGURACAO_OTIMA = "configuracao_otima"
CHAVE_RESULTADO_FINAL = "resultado_final"
CHAVE_HISTORICO_AVALIACOES = "historico_avaliacoes"
CHAVE_RESULTADO_EVOLUCAO = "resultado_evolucao"
CHAVE_TEMPO_OTIMIZACAO = "tempo_otimizacao"

CHAVE_AVALIACAO = "avaliacao"
CHAVE_PERDA_AVALIACAO = "perda_validacao"
CHAVE_ACURACIA_AVALIACAO = "acuracia_validacao"
CHAVE_F1_AVALIACAO = "f1_macro_validacao"
CHAVE_TEMPO_AVALIACAO = "tempo_segundos"


class OtimizadorCNN:
    __ultimoErro = None

    __limites = (
        (-4.0,-2.3),
        (-6.0,-2.5),
        (0.0,0.5),
        (0.0,3.0),
        (0.0,3.0),
        (0.0,3.0)
    )

    __canaisDisponiveis = (8,16,24,32)
    __neuroniosDisponiveis = (32,64,96,128)
    __tamanhosLoteDisponiveis = (32,64,128,256)

    @staticmethod
    def __registrarErro(codigo:int,mensagem:str,origem:str,detalhe:str | None=None) -> None:
        OtimizadorCNN.__ultimoErro = ErroSistema.criar(codigo,mensagem,origem,detalhe)

    @staticmethod
    def __limparUltimoErro() -> None:
        OtimizadorCNN.__ultimoErro = None

    @staticmethod
    def getUltimoErro() -> ErroSistema | None:
        return OtimizadorCNN.__ultimoErro

    @staticmethod
    def ultimaExecucaoDeuErro() -> bool:
        if(OtimizadorCNN.__ultimoErro is None):
            return False

        else:
            return True

    @staticmethod
    def getLimites() -> tuple[tuple[float,float],...]:
        return OtimizadorCNN.__limites

    @staticmethod
    def criarConfiguracaoAPartirDoVetor(vetorVariaveis:np.ndarray | list[float] | tuple[float,...]) -> ConfiguracaoCNN | None:
        OtimizadorCNN.__limparUltimoErro()

        try:
            vetorVariaveis = np.asarray(vetorVariaveis,dtype=np.float64)

            if(vetorVariaveis.shape != (6,)):
                OtimizadorCNN.__registrarErro(CODIGO_ERRO_VETOR_OTIMIZACAO_INVALIDO,"O vetor de otimização deve possuir seis elementos","OtimizadorCNN.criarConfiguracaoAPartirDoVetor")
                return None

            elif(not np.all(np.isfinite(vetorVariaveis))):
                OtimizadorCNN.__registrarErro(CODIGO_ERRO_VETOR_OTIMIZACAO_INVALIDO,"O vetor de otimização contém valores inválidos","OtimizadorCNN.criarConfiguracaoAPartirDoVetor")
                return None

            else:
                for indiceVariavel in range(vetorVariaveis.size):
                    limiteInferior = OtimizadorCNN.__limites[indiceVariavel][0]
                    limiteSuperior = OtimizadorCNN.__limites[indiceVariavel][1]

                    if(vetorVariaveis[indiceVariavel] < limiteInferior or vetorVariaveis[indiceVariavel] > limiteSuperior):
                        OtimizadorCNN.__registrarErro(CODIGO_ERRO_VETOR_OTIMIZACAO_INVALIDO,"Uma variável está fora de seu intervalo permitido","OtimizadorCNN.criarConfiguracaoAPartirDoVetor",str(indiceVariavel))
                        return None

                taxaAprendizado = float(10.0 ** vetorVariaveis[0])
                decaimentoPesos = float(10.0 ** vetorVariaveis[1])
                dropout = float(vetorVariaveis[2])

                indiceCanais = int(np.rint(vetorVariaveis[3]))
                indiceNeuronios = int(np.rint(vetorVariaveis[4]))
                indiceTamanhoLote = int(np.rint(vetorVariaveis[5]))

                indiceCanais = min(max(indiceCanais,0),len(OtimizadorCNN.__canaisDisponiveis) - 1)
                indiceNeuronios = min(max(indiceNeuronios,0),len(OtimizadorCNN.__neuroniosDisponiveis) - 1)
                indiceTamanhoLote = min(max(indiceTamanhoLote,0),len(OtimizadorCNN.__tamanhosLoteDisponiveis) - 1)

                canaisPrimeiraCamada = OtimizadorCNN.__canaisDisponiveis[indiceCanais]
                canaisSegundaCamada = canaisPrimeiraCamada * 2
                canaisTerceiraCamada = canaisPrimeiraCamada * 4

                neuroniosCamadaDensa = OtimizadorCNN.__neuroniosDisponiveis[indiceNeuronios]
                tamanhoLote = OtimizadorCNN.__tamanhosLoteDisponiveis[indiceTamanhoLote]

                configuracao = ConfiguracaoCNN.criar(canaisPrimeiraCamada,canaisSegundaCamada,canaisTerceiraCamada,16,neuroniosCamadaDensa,dropout,tamanhoLote,taxaAprendizado,decaimentoPesos)

                if(configuracao is None):
                    OtimizadorCNN.__registrarErro(CODIGO_ERRO_VETOR_OTIMIZACAO_INVALIDO,"Não foi possível criar a configuração da CNN","OtimizadorCNN.criarConfiguracaoAPartirDoVetor")
                    return None

                else:
                    return configuracao

        except Exception as excecao:
            OtimizadorCNN.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado durante a conversão do vetor de otimização","OtimizadorCNN.criarConfiguracaoAPartirDoVetor",str(excecao))
            return None

    @staticmethod
    def __avaliarCandidato(vetorVariaveis:np.ndarray,matrizDados:np.ndarray,vetorClasses:np.ndarray,indicesTreino:np.ndarray,indicesValidacao:np.ndarray,historicoAvaliacoes:list[dict[str,object]],maximoEpocas:int,paciencia:int,usarCuda:bool,exibirProgresso:bool) -> float:
        numeroAvaliacao = len(historicoAvaliacoes) + 1
        configuracao = OtimizadorCNN.criarConfiguracaoAPartirDoVetor(vetorVariaveis)

        if(configuracao is None):
            historicoAvaliacoes.append({
                CHAVE_AVALIACAO: numeroAvaliacao,
                CHAVE_PERDA_AVALIACAO: OTIMIZACAO_CNN_PENALIDADE,
                CHAVE_ACURACIA_AVALIACAO: 0.0,
                CHAVE_F1_AVALIACAO: 0.0,
                CHAVE_TEMPO_AVALIACAO: 0.0,
                CHAVE_CONFIGURACAO: None
            })

            return OTIMIZACAO_CNN_PENALIDADE

        resultado = TreinadorCNN.treinar(matrizDados,vetorClasses,indicesTreino,indicesValidacao,"temporal",maximoEpocas,paciencia,usarCuda,False,configuracao)

        if(resultado is None):
            historicoAvaliacoes.append({
                CHAVE_AVALIACAO: numeroAvaliacao,
                CHAVE_PERDA_AVALIACAO: OTIMIZACAO_CNN_PENALIDADE,
                CHAVE_ACURACIA_AVALIACAO: 0.0,
                CHAVE_F1_AVALIACAO: 0.0,
                CHAVE_TEMPO_AVALIACAO: 0.0,
                CHAVE_CONFIGURACAO: configuracao
            })

            if(exibirProgresso):
                print("Avaliação %d | configuração rejeitada pelo treinador" % numeroAvaliacao)

            return OTIMIZACAO_CNN_PENALIDADE

        perdaValidacao = float(resultado[CHAVE_PERDA_FINAL])
        acuraciaValidacao = float(resultado[CHAVE_ACURACIA_FINAL])
        f1Validacao = float(resultado[CHAVE_F1_FINAL])
        tempoAvaliacao = float(resultado[CHAVE_TEMPO_TREINAMENTO])

        historicoAvaliacoes.append({
            CHAVE_AVALIACAO: numeroAvaliacao,
            CHAVE_PERDA_AVALIACAO: perdaValidacao,
            CHAVE_ACURACIA_AVALIACAO: acuraciaValidacao,
            CHAVE_F1_AVALIACAO: f1Validacao,
            CHAVE_TEMPO_AVALIACAO: tempoAvaliacao,
            CHAVE_CONFIGURACAO: configuracao
        })

        if(exibirProgresso):
            dicionarioConfiguracao = configuracao.paraDicionario()

            print(
                "Avaliação %d | perda %.6f | acurácia %.4f | F1 %.4f | filtros %d-%d-%d | densa %d | lote %d | dropout %.4f | taxa %.8f | regularização %.8f"
                %
                (
                    numeroAvaliacao,
                    perdaValidacao,
                    acuraciaValidacao,
                    f1Validacao,
                    dicionarioConfiguracao["canais_primeira_camada"],
                    dicionarioConfiguracao["canais_segunda_camada"],
                    dicionarioConfiguracao["canais_terceira_camada"],
                    dicionarioConfiguracao["neuronios_camada_densa"],
                    dicionarioConfiguracao["tamanho_lote"],
                    dicionarioConfiguracao["dropout"],
                    dicionarioConfiguracao["taxa_aprendizado"],
                    dicionarioConfiguracao["decaimento_pesos"]
                )
            )

        return perdaValidacao

    @staticmethod
    def otimizar(matrizDados:np.ndarray,vetorClasses:np.ndarray,indicesTreino:np.ndarray,indicesValidacao:np.ndarray,maximoIteracoes:int=OTIMIZACAO_CNN_MAXIMO_ITERACOES,tamanhoPopulacao:int=OTIMIZACAO_CNN_TAMANHO_POPULACAO,maximoEpocasAvaliacao:int=OTIMIZACAO_CNN_MAXIMO_EPOCAS_AVALIACAO,pacienciaAvaliacao:int=OTIMIZACAO_CNN_PACIENCIA_AVALIACAO,usarCuda:bool=True,exibirProgresso:bool=True) -> dict[str,object] | None:
        OtimizadorCNN.__limparUltimoErro()

        try:
            matrizDados = np.asarray(matrizDados)
            vetorClasses = np.asarray(vetorClasses)
            indicesTreino = np.asarray(indicesTreino)
            indicesValidacao = np.asarray(indicesValidacao)

            if(not isinstance(maximoIteracoes,int) or isinstance(maximoIteracoes,bool) or maximoIteracoes < 0):
                OtimizadorCNN.__registrarErro(CODIGO_ERRO_OTIMIZACAO_CNN,"A quantidade de iterações é inválida","OtimizadorCNN.otimizar")
                return None

            elif(not isinstance(tamanhoPopulacao,int) or isinstance(tamanhoPopulacao,bool) or tamanhoPopulacao < 1):
                OtimizadorCNN.__registrarErro(CODIGO_ERRO_OTIMIZACAO_CNN,"O tamanho da população é inválido","OtimizadorCNN.otimizar")
                return None

            elif(not isinstance(maximoEpocasAvaliacao,int) or isinstance(maximoEpocasAvaliacao,bool) or maximoEpocasAvaliacao < 1):
                OtimizadorCNN.__registrarErro(CODIGO_ERRO_OTIMIZACAO_CNN,"A quantidade de épocas por avaliação é inválida","OtimizadorCNN.otimizar")
                return None

            elif(not isinstance(pacienciaAvaliacao,int) or isinstance(pacienciaAvaliacao,bool) or pacienciaAvaliacao < 1):
                OtimizadorCNN.__registrarErro(CODIGO_ERRO_OTIMIZACAO_CNN,"A paciência das avaliações é inválida","OtimizadorCNN.otimizar")
                return None

            elif(not isinstance(usarCuda,bool) or not isinstance(exibirProgresso,bool)):
                OtimizadorCNN.__registrarErro(CODIGO_ERRO_OTIMIZACAO_CNN,"Um parâmetro booleano é inválido","OtimizadorCNN.otimizar")
                return None

            else:
                historicoAvaliacoes = []

                funcaoObjetivo = partial(
                    OtimizadorCNN.__avaliarCandidato,
                    matrizDados=matrizDados,
                    vetorClasses=vetorClasses,
                    indicesTreino=indicesTreino,
                    indicesValidacao=indicesValidacao,
                    historicoAvaliacoes=historicoAvaliacoes,
                    maximoEpocas=maximoEpocasAvaliacao,
                    paciencia=pacienciaAvaliacao,
                    usarCuda=usarCuda,
                    exibirProgresso=exibirProgresso
                )

                vetorConfiguracaoPadrao = np.asarray([-3.0,-4.0,0.20,1.0,1.0,2.0],dtype=np.float64)

                instanteInicial = time.perf_counter()

                resultadoEvolucao = differential_evolution(
                    funcaoObjetivo,
                    bounds=OtimizadorCNN.__limites,
                    strategy="best1bin",
                    maxiter=maximoIteracoes,
                    popsize=tamanhoPopulacao,
                    tol=0.0,
                    mutation=(0.5,1.0),
                    recombination=0.7,
                    seed=SEMENTE_ALEATORIA,
                    callback=None,
                    disp=False,
                    polish=False,
                    init="latinhypercube",
                    atol=0.0,
                    updating="immediate",
                    workers=1,
                    x0=vetorConfiguracaoPadrao
                )

                configuracaoOtima = OtimizadorCNN.criarConfiguracaoAPartirDoVetor(resultadoEvolucao.x)

                if(configuracaoOtima is None):
                    OtimizadorCNN.__registrarErro(CODIGO_ERRO_OTIMIZACAO_CNN,"O otimizador não produziu uma configuração válida","OtimizadorCNN.otimizar")
                    return None

                if(exibirProgresso):
                    print()
                    print("Treinando definitivamente a melhor configuração.")

                resultadoFinal = TreinadorCNN.treinar(matrizDados,vetorClasses,indicesTreino,indicesValidacao,"temporal_otimizada",CNN_MAXIMO_EPOCAS,CNN_PACIENCIA,usarCuda,exibirProgresso,configuracaoOtima)

                if(resultadoFinal is None):
                    erroTreinamento = TreinadorCNN.getUltimoErro()

                    if(erroTreinamento is None):
                        OtimizadorCNN.__registrarErro(CODIGO_ERRO_OTIMIZACAO_CNN,"Não foi possível treinar a configuração ótima","OtimizadorCNN.otimizar")

                    else:
                        OtimizadorCNN.__ultimoErro = erroTreinamento

                    return None

                tempoOtimizacao = time.perf_counter() - instanteInicial

                return {
                    CHAVE_CONFIGURACAO_OTIMA: configuracaoOtima,
                    CHAVE_RESULTADO_FINAL: resultadoFinal,
                    CHAVE_HISTORICO_AVALIACOES: historicoAvaliacoes,
                    CHAVE_RESULTADO_EVOLUCAO: resultadoEvolucao,
                    CHAVE_TEMPO_OTIMIZACAO: float(tempoOtimizacao)
                }

        except Exception as excecao:
            OtimizadorCNN.__registrarErro(CODIGO_ERRO_OTIMIZACAO_CNN,"Não foi possível executar a otimização da CNN","OtimizadorCNN.otimizar",str(excecao))
            return None

    @staticmethod
    def __validarResultadoOtimizacao(resultadoOtimizacao:dict[str,object]) -> bool:
        try:
            if(not isinstance(resultadoOtimizacao,dict)):
                OtimizadorCNN.__registrarErro(CODIGO_ERRO_RESULTADO_OTIMIZACAO_INVALIDO,"O resultado da otimização não é um dicionário","OtimizadorCNN.__validarResultadoOtimizacao")
                return False

            else:
                chavesObrigatorias = (
                    CHAVE_CONFIGURACAO_OTIMA,
                    CHAVE_RESULTADO_FINAL,
                    CHAVE_HISTORICO_AVALIACOES,
                    CHAVE_RESULTADO_EVOLUCAO,
                    CHAVE_TEMPO_OTIMIZACAO
                )

                for chave in chavesObrigatorias:
                    if(chave not in resultadoOtimizacao):
                        OtimizadorCNN.__registrarErro(CODIGO_ERRO_RESULTADO_OTIMIZACAO_INVALIDO,"Uma chave obrigatória não foi encontrada","OtimizadorCNN.__validarResultadoOtimizacao",chave)
                        return False

                if(not isinstance(resultadoOtimizacao[CHAVE_CONFIGURACAO_OTIMA],ConfiguracaoCNN)):
                    OtimizadorCNN.__registrarErro(CODIGO_ERRO_RESULTADO_OTIMIZACAO_INVALIDO,"A configuração ótima é inválida","OtimizadorCNN.__validarResultadoOtimizacao")
                    return False

                elif(not isinstance(resultadoOtimizacao[CHAVE_RESULTADO_FINAL],dict)):
                    OtimizadorCNN.__registrarErro(CODIGO_ERRO_RESULTADO_OTIMIZACAO_INVALIDO,"O resultado final é inválido","OtimizadorCNN.__validarResultadoOtimizacao")
                    return False

                elif(not isinstance(resultadoOtimizacao[CHAVE_HISTORICO_AVALIACOES],list)):
                    OtimizadorCNN.__registrarErro(CODIGO_ERRO_RESULTADO_OTIMIZACAO_INVALIDO,"O histórico de avaliações é inválido","OtimizadorCNN.__validarResultadoOtimizacao")
                    return False

                elif(len(resultadoOtimizacao[CHAVE_HISTORICO_AVALIACOES]) == 0):
                    OtimizadorCNN.__registrarErro(CODIGO_ERRO_RESULTADO_OTIMIZACAO_INVALIDO,"O histórico de avaliações está vazio","OtimizadorCNN.__validarResultadoOtimizacao")
                    return False

                else:
                    return True

        except Exception as excecao:
            OtimizadorCNN.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado durante a validação do resultado da otimização","OtimizadorCNN.__validarResultadoOtimizacao",str(excecao))
            return False

    @staticmethod
    def salvarResultados(resultadoOtimizacao:dict[str,object],diretorioResultados:Path | str=DIRETORIO_RESULTADOS,diretorioGraficos:Path | str=DIRETORIO_GRAFICOS,diretorioModelos:Path | str=DIRETORIO_MODELOS) -> list[Path] | None:
        OtimizadorCNN.__limparUltimoErro()

        try:
            if(not OtimizadorCNN.__validarResultadoOtimizacao(resultadoOtimizacao)):
                return None

            diretorioResultados = Path(diretorioResultados)
            diretorioGraficos = Path(diretorioGraficos)
            diretorioModelos = Path(diretorioModelos)

            diretorioResultados.mkdir(parents=True,exist_ok=True)
            diretorioGraficos.mkdir(parents=True,exist_ok=True)
            diretorioModelos.mkdir(parents=True,exist_ok=True)

            configuracaoOtima = resultadoOtimizacao[CHAVE_CONFIGURACAO_OTIMA]
            resultadoFinal = resultadoOtimizacao[CHAVE_RESULTADO_FINAL]
            historicoAvaliacoes = resultadoOtimizacao[CHAVE_HISTORICO_AVALIACOES]
            resultadoEvolucao = resultadoOtimizacao[CHAVE_RESULTADO_EVOLUCAO]
            tempoOtimizacao = resultadoOtimizacao[CHAVE_TEMPO_OTIMIZACAO]

            listaDeArquivos = []

            caminhoHistoricoOtimizacao = diretorioResultados / "otimizacao_cnn_historico.csv"

            with open(caminhoHistoricoOtimizacao,"w",encoding="utf-8",newline="") as arquivoCsv:
                escritor = csv.writer(arquivoCsv)

                escritor.writerow([
                    "avaliacao",
                    "perda_validacao",
                    "acuracia_validacao",
                    "f1_macro_validacao",
                    "tempo_segundos",
                    "canais_primeira_camada",
                    "canais_segunda_camada",
                    "canais_terceira_camada",
                    "neuronios_camada_densa",
                    "dropout",
                    "tamanho_lote",
                    "taxa_aprendizado",
                    "decaimento_pesos"
                ])

                for itemHistorico in historicoAvaliacoes:
                    configuracao = itemHistorico[CHAVE_CONFIGURACAO]

                    if(configuracao is None):
                        dicionarioConfiguracao = {
                            "canais_primeira_camada": 0,
                            "canais_segunda_camada": 0,
                            "canais_terceira_camada": 0,
                            "neuronios_camada_densa": 0,
                            "dropout": 0.0,
                            "tamanho_lote": 0,
                            "taxa_aprendizado": 0.0,
                            "decaimento_pesos": 0.0
                        }

                    else:
                        dicionarioConfiguracao = configuracao.paraDicionario()

                    escritor.writerow([
                        int(itemHistorico[CHAVE_AVALIACAO]),
                        "%.12f" % float(itemHistorico[CHAVE_PERDA_AVALIACAO]),
                        "%.12f" % float(itemHistorico[CHAVE_ACURACIA_AVALIACAO]),
                        "%.12f" % float(itemHistorico[CHAVE_F1_AVALIACAO]),
                        "%.6f" % float(itemHistorico[CHAVE_TEMPO_AVALIACAO]),
                        dicionarioConfiguracao["canais_primeira_camada"],
                        dicionarioConfiguracao["canais_segunda_camada"],
                        dicionarioConfiguracao["canais_terceira_camada"],
                        dicionarioConfiguracao["neuronios_camada_densa"],
                        "%.12f" % dicionarioConfiguracao["dropout"],
                        dicionarioConfiguracao["tamanho_lote"],
                        "%.12f" % dicionarioConfiguracao["taxa_aprendizado"],
                        "%.12f" % dicionarioConfiguracao["decaimento_pesos"]
                    ])

            listaDeArquivos.append(caminhoHistoricoOtimizacao)

            caminhoConfiguracao = diretorioResultados / "otimizacao_cnn_melhor_configuracao.csv"
            dicionarioConfiguracaoOtima = configuracaoOtima.paraDicionario()

            with open(caminhoConfiguracao,"w",encoding="utf-8",newline="") as arquivoCsv:
                escritor = csv.writer(arquivoCsv)
                escritor.writerow(["parametro","valor"])

                for nomeParametro in dicionarioConfiguracaoOtima:
                    escritor.writerow([nomeParametro,dicionarioConfiguracaoOtima[nomeParametro]])

                escritor.writerow(["avaliacoes",int(resultadoEvolucao.nfev)])
                escritor.writerow(["valor_objetivo_busca","%.12f" % float(resultadoEvolucao.fun)])
                escritor.writerow(["tempo_total_segundos","%.6f" % float(tempoOtimizacao)])

            listaDeArquivos.append(caminhoConfiguracao)

            caminhoResumoFinal = diretorioResultados / "cnn_otimizada_resumo_validacao.csv"

            with open(caminhoResumoFinal,"w",encoding="utf-8",newline="") as arquivoCsv:
                escritor = csv.writer(arquivoCsv)

                escritor.writerow([
                    "representacao",
                    "melhor_epoca",
                    "epocas_executadas",
                    "perda_validacao",
                    "acuracia_validacao",
                    "f1_macro_validacao",
                    "tempo_treinamento_segundos",
                    "tempo_otimizacao_segundos",
                    "quantidade_parametros",
                    "dispositivo",
                    "memoria_maxima_mb"
                ])

                escritor.writerow([
                    "temporal_otimizada",
                    int(resultadoFinal[CHAVE_MELHOR_EPOCA]),
                    int(np.asarray(resultadoFinal[CHAVE_EPOCAS]).size),
                    "%.12f" % float(resultadoFinal[CHAVE_PERDA_FINAL]),
                    "%.12f" % float(resultadoFinal[CHAVE_ACURACIA_FINAL]),
                    "%.12f" % float(resultadoFinal[CHAVE_F1_FINAL]),
                    "%.6f" % float(resultadoFinal[CHAVE_TEMPO_TREINAMENTO]),
                    "%.6f" % float(tempoOtimizacao),
                    int(resultadoFinal[CHAVE_QUANTIDADE_PARAMETROS]),
                    resultadoFinal[CHAVE_DISPOSITIVO],
                    "%.6f" % float(resultadoFinal[CHAVE_MEMORIA_MAXIMA_MB])
                ])

            listaDeArquivos.append(caminhoResumoFinal)

            caminhoHistoricoFinal = diretorioResultados / "cnn_otimizada_historico_temporal.csv"

            vetorEpocas = np.asarray(resultadoFinal[CHAVE_EPOCAS])
            vetorPerdasTreino = np.asarray(resultadoFinal[CHAVE_PERDAS_TREINO])
            vetorPerdasValidacao = np.asarray(resultadoFinal[CHAVE_PERDAS_VALIDACAO])

            with open(caminhoHistoricoFinal,"w",encoding="utf-8",newline="") as arquivoCsv:
                escritor = csv.writer(arquivoCsv)
                escritor.writerow(["epoca","perda_treino","perda_validacao"])

                for indiceEpoca in range(vetorEpocas.size):
                    escritor.writerow([
                        int(vetorEpocas[indiceEpoca]),
                        "%.12f" % vetorPerdasTreino[indiceEpoca],
                        "%.12f" % vetorPerdasValidacao[indiceEpoca]
                    ])

            listaDeArquivos.append(caminhoHistoricoFinal)

            matrizConfusao = np.asarray(resultadoFinal[CHAVE_MATRIZ_CONFUSAO])
            caminhoMatrizConfusao = diretorioResultados / "cnn_otimizada_matriz_confusao_temporal.csv"

            with open(caminhoMatrizConfusao,"w",encoding="utf-8",newline="") as arquivoCsv:
                escritor = csv.writer(arquivoCsv)
                escritor.writerow(["classe_real","predita_0","predita_1","predita_2","predita_3"])

                for classeReal in range(QUANTIDADE_AMBIENTES):
                    linha = [classeReal]

                    for classePredita in range(QUANTIDADE_AMBIENTES):
                        linha.append(int(matrizConfusao[classeReal,classePredita]))

                    escritor.writerow(linha)

            listaDeArquivos.append(caminhoMatrizConfusao)

            caminhoResumoPadrao = diretorioResultados / "cnn_resumo_validacao.csv"

            if(not caminhoResumoPadrao.exists()):
                OtimizadorCNN.__registrarErro(CODIGO_ERRO_SALVAMENTO_OTIMIZACAO,"O resumo da CNN padrão não foi encontrado","OtimizadorCNN.salvarResultados",str(caminhoResumoPadrao))
                return None

            with open(caminhoResumoPadrao,"r",encoding="utf-8",newline="") as arquivoCsv:
                leitor = csv.DictReader(arquivoCsv)
                linhaPadrao = next(leitor,None)

            if(linhaPadrao is None):
                OtimizadorCNN.__registrarErro(CODIGO_ERRO_SALVAMENTO_OTIMIZACAO,"O resumo da CNN padrão está vazio","OtimizadorCNN.salvarResultados")
                return None

            caminhoComparacao = diretorioResultados / "comparacao_cnn_padrao_otimizada_validacao.csv"

            with open(caminhoComparacao,"w",encoding="utf-8",newline="") as arquivoCsv:
                escritor = csv.writer(arquivoCsv)

                escritor.writerow([
                    "modelo",
                    "perda_validacao",
                    "acuracia_validacao",
                    "f1_macro_validacao",
                    "tempo_treinamento_segundos",
                    "quantidade_parametros"
                ])

                escritor.writerow([
                    "CNN padrão",
                    linhaPadrao["perda_validacao"],
                    linhaPadrao["acuracia_validacao"],
                    linhaPadrao["f1_macro_validacao"],
                    linhaPadrao["tempo_segundos"],
                    linhaPadrao["quantidade_parametros"]
                ])

                escritor.writerow([
                    "CNN otimizada",
                    "%.12f" % float(resultadoFinal[CHAVE_PERDA_FINAL]),
                    "%.12f" % float(resultadoFinal[CHAVE_ACURACIA_FINAL]),
                    "%.12f" % float(resultadoFinal[CHAVE_F1_FINAL]),
                    "%.6f" % float(resultadoFinal[CHAVE_TEMPO_TREINAMENTO]),
                    int(resultadoFinal[CHAVE_QUANTIDADE_PARAMETROS])
                ])

            listaDeArquivos.append(caminhoComparacao)

            caminhoModelo = diretorioModelos / "cnn_temporal_otimizada.pt"

            pacoteModelo = {
                "estado_modelo": resultadoFinal[CHAVE_MODELO].state_dict(),
                "configuracao": configuracaoOtima.paraDicionario(),
                "quantidade_classes": QUANTIDADE_AMBIENTES,
                "quantidade_pontos": int(resultadoFinal[CHAVE_QUANTIDADE_PONTOS])
            }

            torch.save(pacoteModelo,caminhoModelo)
            listaDeArquivos.append(caminhoModelo)

            caminhoGraficoConvergencia = diretorioGraficos / "otimizacao_cnn_convergencia.png"

            listaDePerdas = []

            for itemHistorico in historicoAvaliacoes:
                listaDePerdas.append(float(itemHistorico[CHAVE_PERDA_AVALIACAO]))

            vetorPerdas = np.asarray(listaDePerdas,dtype=np.float64)
            vetorMelhoresPerdas = np.minimum.accumulate(vetorPerdas)
            vetorAvaliacoes = np.arange(1,vetorPerdas.size + 1)

            plt.figure(figsize=(10,6))
            plt.plot(vetorAvaliacoes,vetorPerdas,label="Perda da avaliação")
            plt.plot(vetorAvaliacoes,vetorMelhoresPerdas,label="Melhor perda acumulada")
            plt.xlabel("Avaliação da função objetivo")
            plt.ylabel("Perda de validação")
            plt.title("Convergência da evolução diferencial")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(caminhoGraficoConvergencia,dpi=300,bbox_inches="tight")
            plt.close()

            listaDeArquivos.append(caminhoGraficoConvergencia)

            caminhoGraficoTreinamento = diretorioGraficos / "cnn_otimizada_curva_temporal.png"

            plt.figure(figsize=(9,6))
            plt.plot(vetorEpocas,vetorPerdasTreino,label="Treino")
            plt.plot(vetorEpocas,vetorPerdasValidacao,label="Validação")
            plt.axvline(int(resultadoFinal[CHAVE_MELHOR_EPOCA]),linestyle="--",label="Melhor época")
            plt.xlabel("Época")
            plt.ylabel("Entropia cruzada")
            plt.title("Treinamento da CNN 1D otimizada")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(caminhoGraficoTreinamento,dpi=300,bbox_inches="tight")
            plt.close()

            listaDeArquivos.append(caminhoGraficoTreinamento)

            caminhoGraficoMatriz = diretorioGraficos / "cnn_otimizada_matriz_confusao_temporal.png"
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
            plt.title("Matriz de confusão da CNN 1D otimizada")
            plt.tight_layout()
            plt.savefig(caminhoGraficoMatriz,dpi=300,bbox_inches="tight")
            plt.close()

            listaDeArquivos.append(caminhoGraficoMatriz)

            caminhoGraficoComparacao = diretorioGraficos / "comparacao_cnn_padrao_otimizada_validacao.png"

            nomesModelos = ["CNN padrão","CNN otimizada"]
            acuracias = [
                float(linhaPadrao["acuracia_validacao"]),
                float(resultadoFinal[CHAVE_ACURACIA_FINAL])
            ]
            valoresF1 = [
                float(linhaPadrao["f1_macro_validacao"]),
                float(resultadoFinal[CHAVE_F1_FINAL])
            ]

            posicoes = np.arange(2)
            larguraBarra = 0.35

            plt.figure(figsize=(8,6))
            plt.bar(posicoes - larguraBarra / 2.0,acuracias,width=larguraBarra,label="Acurácia")
            plt.bar(posicoes + larguraBarra / 2.0,valoresF1,width=larguraBarra,label="F1 macro")
            plt.xticks(posicoes,nomesModelos)
            plt.ylim(0.0,1.0)
            plt.xlabel("Modelo")
            plt.ylabel("Métrica de validação")
            plt.title("Efeito da otimização dos hiperparâmetros")
            plt.legend()
            plt.grid(True,axis="y")
            plt.tight_layout()
            plt.savefig(caminhoGraficoComparacao,dpi=300,bbox_inches="tight")
            plt.close()

            listaDeArquivos.append(caminhoGraficoComparacao)

            return listaDeArquivos

        except Exception as excecao:
            OtimizadorCNN.__registrarErro(CODIGO_ERRO_SALVAMENTO_OTIMIZACAO,"Não foi possível salvar os resultados da otimização","OtimizadorCNN.salvarResultados",str(excecao))
            return None