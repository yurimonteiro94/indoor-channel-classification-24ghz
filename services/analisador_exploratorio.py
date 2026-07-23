from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from model.entidades.erro_sistema import ErroSistema
from services.constantes import CAMINHO_BASE_PREPARADA
from services.constantes import CAMINHO_BASE_PROCESSADA
from services.constantes import CHAVE_ATRASOS
from services.constantes import CHAVE_CLASSES
from services.constantes import CHAVE_FREQUENCIAS
from services.constantes import CHAVE_GRUPOS
from services.constantes import CHAVE_INDICES_TESTE
from services.constantes import CHAVE_INDICES_TREINO
from services.constantes import CHAVE_INDICES_VALIDACAO
from services.constantes import CHAVE_REPRESENTACAO_CARTESIANA
from services.constantes import CHAVE_S21
from services.constantes import CODIGO_ERRO_ANALISE_EXPLORATORIA
from services.constantes import CODIGO_ERRO_BASES_INCOMPATIVEIS
from services.constantes import CODIGO_ERRO_DESCONHECIDO
from services.constantes import CODIGO_ERRO_SALVAMENTO_ANALISE
from services.constantes import DICIONARIO_NOMES_POR_CLASSE
from services.constantes import DIRETORIO_GRAFICOS
from services.constantes import DIRETORIO_RESULTADOS
from services.constantes import EPSILON_MAGNITUDE
from services.constantes import QUANTIDADE_AMBIENTES
from services.constantes import QUANTIDADE_PONTOS_FREQUENCIA
from services.constantes import SEMENTE_ALEATORIA
from services.preparador_experimentos import PreparadorExperimentos
from services.processador_base import ProcessadorBase


CHAVE_FREQUENCIAS_GHZ = "frequencias_ghz"
CHAVE_ATRASOS_US = "atrasos_us"
CHAVE_CONTAGENS_MEDICOES = "contagens_medicoes"
CHAVE_CONTAGENS_GRUPOS = "contagens_grupos"
CHAVE_MAGNITUDE_MEDIA_DB = "magnitude_media_db"
CHAVE_MAGNITUDE_DESVIO_DB = "magnitude_desvio_db"
CHAVE_TEMPORAL_MEDIA_DB = "temporal_media_db"
CHAVE_COORDENADAS_PCA = "coordenadas_pca"
CHAVE_CLASSES_PCA = "classes_pca"
CHAVE_VARIANCIA_PCA = "variancia_pca"
CHAVE_ERROS_REPETICOES = "erros_repeticoes"
CHAVE_CLASSES_ESTABILIDADE = "classes_estabilidade"
CHAVE_RESUMO_ESTABILIDADE = "resumo_estabilidade"


class AnalisadorExploratorio:
    __ultimoErro = None

    @staticmethod
    def __registrarErro(codigo:int,mensagem:str,origem:str,detalhe:str | None=None) -> None:
        AnalisadorExploratorio.__ultimoErro = ErroSistema.criar(codigo,mensagem,origem,detalhe)

    @staticmethod
    def __limparUltimoErro() -> None:
        AnalisadorExploratorio.__ultimoErro = None

    @staticmethod
    def __copiarErro(erro:ErroSistema | None,codigo:int,mensagem:str,origem:str) -> None:
        if(erro is None):
            AnalisadorExploratorio.__registrarErro(codigo,mensagem,origem)

        else:
            AnalisadorExploratorio.__ultimoErro = erro

    @staticmethod
    def getUltimoErro() -> ErroSistema | None:
        return AnalisadorExploratorio.__ultimoErro

    @staticmethod
    def ultimaExecucaoDeuErro() -> bool:
        if(AnalisadorExploratorio.__ultimoErro is None):
            return False

        else:
            return True

    @staticmethod
    def __validarBases(dicionarioBase:dict[str,np.ndarray],dicionarioBasePreparada:dict[str,np.ndarray]) -> bool:
        try:
            if(not ProcessadorBase.validarBase(dicionarioBase,False)):
                AnalisadorExploratorio.__copiarErro(ProcessadorBase.getUltimoErro(),CODIGO_ERRO_BASES_INCOMPATIVEIS,"A base processada é inválida","AnalisadorExploratorio.__validarBases")
                return False

            elif(not PreparadorExperimentos.validarBasePreparada(dicionarioBasePreparada)):
                AnalisadorExploratorio.__copiarErro(PreparadorExperimentos.getUltimoErro(),CODIGO_ERRO_BASES_INCOMPATIVEIS,"A base preparada é inválida","AnalisadorExploratorio.__validarBases")
                return False

            elif(not np.array_equal(dicionarioBase[CHAVE_CLASSES],dicionarioBasePreparada[CHAVE_CLASSES])):
                AnalisadorExploratorio.__registrarErro(CODIGO_ERRO_BASES_INCOMPATIVEIS,"Os vetores de classes das duas bases são diferentes","AnalisadorExploratorio.__validarBases")
                return False

            elif(not np.array_equal(dicionarioBase[CHAVE_GRUPOS],dicionarioBasePreparada[CHAVE_GRUPOS])):
                AnalisadorExploratorio.__registrarErro(CODIGO_ERRO_BASES_INCOMPATIVEIS,"Os vetores de grupos das duas bases são diferentes","AnalisadorExploratorio.__validarBases")
                return False

            elif(not np.allclose(dicionarioBase[CHAVE_FREQUENCIAS],dicionarioBasePreparada[CHAVE_FREQUENCIAS],rtol=0.0,atol=1.0)):
                AnalisadorExploratorio.__registrarErro(CODIGO_ERRO_BASES_INCOMPATIVEIS,"Os vetores de frequências das duas bases são diferentes","AnalisadorExploratorio.__validarBases")
                return False

            else:
                return True

        except Exception as excecao:
            AnalisadorExploratorio.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado durante a validação das bases","AnalisadorExploratorio.__validarBases",str(excecao))
            return False

    @staticmethod
    def gerarResultados(dicionarioBase:dict[str,np.ndarray],dicionarioBasePreparada:dict[str,np.ndarray]) -> dict[str,np.ndarray] | None:
        AnalisadorExploratorio.__limparUltimoErro()

        try:
            if(not AnalisadorExploratorio.__validarBases(dicionarioBase,dicionarioBasePreparada)):
                return None

            else:
                matrizS21 = np.asarray(dicionarioBase[CHAVE_S21],dtype=np.float32)
                vetorClasses = np.asarray(dicionarioBase[CHAVE_CLASSES])
                vetorGrupos = np.asarray(dicionarioBase[CHAVE_GRUPOS])
                vetorFrequencias = np.asarray(dicionarioBase[CHAVE_FREQUENCIAS],dtype=np.float64)
                vetorAtrasos = np.asarray(dicionarioBasePreparada[CHAVE_ATRASOS],dtype=np.float64)

                indicesTreino = np.asarray(dicionarioBasePreparada[CHAVE_INDICES_TREINO])
                indicesValidacao = np.asarray(dicionarioBasePreparada[CHAVE_INDICES_VALIDACAO])
                indicesTeste = np.asarray(dicionarioBasePreparada[CHAVE_INDICES_TESTE])

                parteReal = matrizS21[:,:,0].astype(np.complex64)
                parteImaginaria = matrizS21[:,:,1].astype(np.complex64)
                matrizComplexa = parteReal + 1j * parteImaginaria

                matrizMagnitudeLinear = np.abs(matrizComplexa)
                matrizMagnitudeLinear = np.maximum(matrizMagnitudeLinear,EPSILON_MAGNITUDE)
                matrizMagnitudeDb = 20.0 * np.log10(matrizMagnitudeLinear)
                matrizMagnitudeDb = np.asarray(matrizMagnitudeDb,dtype=np.float32)

                matrizContagensMedicoes = np.zeros((3,QUANTIDADE_AMBIENTES),dtype=np.int32)
                matrizContagensGrupos = np.zeros((3,QUANTIDADE_AMBIENTES),dtype=np.int32)

                listaDeIndices = [indicesTreino,indicesValidacao,indicesTeste]

                for indiceSubconjunto in range(3):
                    indicesSubconjunto = listaDeIndices[indiceSubconjunto]

                    for classe in range(QUANTIDADE_AMBIENTES):
                        indicesDaClasse = indicesSubconjunto[vetorClasses[indicesSubconjunto] == classe]
                        matrizContagensMedicoes[indiceSubconjunto,classe] = indicesDaClasse.size
                        matrizContagensGrupos[indiceSubconjunto,classe] = np.unique(vetorGrupos[indicesDaClasse]).size

                matrizMagnitudeMediaDb = np.empty((QUANTIDADE_AMBIENTES,QUANTIDADE_PONTOS_FREQUENCIA),dtype=np.float32)
                matrizMagnitudeDesvioDb = np.empty((QUANTIDADE_AMBIENTES,QUANTIDADE_PONTOS_FREQUENCIA),dtype=np.float32)
                matrizTemporalMediaDb = np.empty((QUANTIDADE_AMBIENTES,QUANTIDADE_PONTOS_FREQUENCIA),dtype=np.float32)

                for classe in range(QUANTIDADE_AMBIENTES):
                    indicesDaClasse = indicesTreino[vetorClasses[indicesTreino] == classe]

                    magnitudeMediaLinear = np.mean(matrizMagnitudeLinear[indicesDaClasse],axis=0,dtype=np.float64)
                    magnitudeMediaLinear = np.maximum(magnitudeMediaLinear,EPSILON_MAGNITUDE)
                    magnitudeMediaDb = 20.0 * np.log10(magnitudeMediaLinear)

                    magnitudeDesvioDb = np.std(matrizMagnitudeDb[indicesDaClasse],axis=0,dtype=np.float64)

                    respostaTemporal = np.fft.ifft(matrizComplexa[indicesDaClasse],axis=1)
                    temporalMediaLinear = np.mean(np.abs(respostaTemporal),axis=0,dtype=np.float64)
                    temporalMediaLinear = np.maximum(temporalMediaLinear,EPSILON_MAGNITUDE)
                    temporalMediaDb = 20.0 * np.log10(temporalMediaLinear)
                    temporalMediaDb = temporalMediaDb - np.max(temporalMediaDb)

                    matrizMagnitudeMediaDb[classe] = magnitudeMediaDb.astype(np.float32)
                    matrizMagnitudeDesvioDb[classe] = magnitudeDesvioDb.astype(np.float32)
                    matrizTemporalMediaDb[classe] = temporalMediaDb.astype(np.float32)

                listaDeErrosRepeticoes = []
                listaDeClassesEstabilidade = []

                gruposTreino = np.unique(vetorGrupos[indicesTreino])

                for grupo in gruposTreino:
                    indicesDoGrupo = indicesTreino[vetorGrupos[indicesTreino] == grupo]
                    perfilMedioDoGrupo = np.mean(matrizMagnitudeDb[indicesDoGrupo],axis=0,dtype=np.float64)
                    diferencas = matrizMagnitudeDb[indicesDoGrupo] - perfilMedioDoGrupo
                    errosDoGrupo = np.sqrt(np.mean(np.square(diferencas),axis=1,dtype=np.float64))
                    classeDoGrupo = int(vetorClasses[indicesDoGrupo[0]])

                    for erroDoGrupo in errosDoGrupo:
                        listaDeErrosRepeticoes.append(float(erroDoGrupo))
                        listaDeClassesEstabilidade.append(classeDoGrupo)

                vetorErrosRepeticoes = np.asarray(listaDeErrosRepeticoes,dtype=np.float32)
                vetorClassesEstabilidade = np.asarray(listaDeClassesEstabilidade,dtype=np.int8)

                matrizResumoEstabilidade = np.empty((QUANTIDADE_AMBIENTES,5),dtype=np.float64)

                for classe in range(QUANTIDADE_AMBIENTES):
                    errosDaClasse = vetorErrosRepeticoes[vetorClassesEstabilidade == classe]

                    matrizResumoEstabilidade[classe,0] = np.mean(errosDaClasse,dtype=np.float64)
                    matrizResumoEstabilidade[classe,1] = np.std(errosDaClasse,dtype=np.float64)
                    matrizResumoEstabilidade[classe,2] = np.median(errosDaClasse)
                    matrizResumoEstabilidade[classe,3] = np.percentile(errosDaClasse,95.0)
                    matrizResumoEstabilidade[classe,4] = errosDaClasse.size

                matrizCartesianaTreino = np.asarray(dicionarioBasePreparada[CHAVE_REPRESENTACAO_CARTESIANA][indicesTreino],dtype=np.float32)
                matrizCartesianaTreino = matrizCartesianaTreino.reshape(matrizCartesianaTreino.shape[0],-1)

                modeloPca = PCA(n_components=2,svd_solver="randomized",random_state=SEMENTE_ALEATORIA)
                coordenadasPca = modeloPca.fit_transform(matrizCartesianaTreino)
                varianciaPca = modeloPca.explained_variance_ratio_

                return {
                    CHAVE_FREQUENCIAS_GHZ: vetorFrequencias / 1000000000.0,
                    CHAVE_ATRASOS_US: vetorAtrasos * 1000000.0,
                    CHAVE_CONTAGENS_MEDICOES: matrizContagensMedicoes,
                    CHAVE_CONTAGENS_GRUPOS: matrizContagensGrupos,
                    CHAVE_MAGNITUDE_MEDIA_DB: matrizMagnitudeMediaDb,
                    CHAVE_MAGNITUDE_DESVIO_DB: matrizMagnitudeDesvioDb,
                    CHAVE_TEMPORAL_MEDIA_DB: matrizTemporalMediaDb,
                    CHAVE_COORDENADAS_PCA: np.asarray(coordenadasPca,dtype=np.float32),
                    CHAVE_CLASSES_PCA: vetorClasses[indicesTreino].astype(np.int8),
                    CHAVE_VARIANCIA_PCA: np.asarray(varianciaPca,dtype=np.float64),
                    CHAVE_ERROS_REPETICOES: vetorErrosRepeticoes,
                    CHAVE_CLASSES_ESTABILIDADE: vetorClassesEstabilidade,
                    CHAVE_RESUMO_ESTABILIDADE: matrizResumoEstabilidade
                }

        except Exception as excecao:
            AnalisadorExploratorio.__registrarErro(CODIGO_ERRO_ANALISE_EXPLORATORIA,"Não foi possível gerar os resultados da análise exploratória","AnalisadorExploratorio.gerarResultados",str(excecao))
            return None

    @staticmethod
    def __validarResultados(dicionarioResultados:dict[str,np.ndarray]) -> bool:
        try:
            if(not isinstance(dicionarioResultados,dict)):
                AnalisadorExploratorio.__registrarErro(CODIGO_ERRO_ANALISE_EXPLORATORIA,"Os resultados não foram fornecidos em um dicionário","AnalisadorExploratorio.__validarResultados")
                return False

            else:
                chavesObrigatorias = (
                    CHAVE_FREQUENCIAS_GHZ,CHAVE_ATRASOS_US,CHAVE_CONTAGENS_MEDICOES,CHAVE_CONTAGENS_GRUPOS,
                    CHAVE_MAGNITUDE_MEDIA_DB,CHAVE_MAGNITUDE_DESVIO_DB,CHAVE_TEMPORAL_MEDIA_DB,
                    CHAVE_COORDENADAS_PCA,CHAVE_CLASSES_PCA,CHAVE_VARIANCIA_PCA,
                    CHAVE_ERROS_REPETICOES,CHAVE_CLASSES_ESTABILIDADE,CHAVE_RESUMO_ESTABILIDADE
                )

                for chave in chavesObrigatorias:
                    if(chave not in dicionarioResultados):
                        AnalisadorExploratorio.__registrarErro(CODIGO_ERRO_ANALISE_EXPLORATORIA,"Uma chave obrigatória não foi encontrada nos resultados","AnalisadorExploratorio.__validarResultados",chave)
                        return False

                if(np.asarray(dicionarioResultados[CHAVE_CONTAGENS_MEDICOES]).shape != (3,QUANTIDADE_AMBIENTES)):
                    AnalisadorExploratorio.__registrarErro(CODIGO_ERRO_ANALISE_EXPLORATORIA,"A matriz de contagens de medições possui formato inválido","AnalisadorExploratorio.__validarResultados")
                    return False

                elif(np.asarray(dicionarioResultados[CHAVE_CONTAGENS_GRUPOS]).shape != (3,QUANTIDADE_AMBIENTES)):
                    AnalisadorExploratorio.__registrarErro(CODIGO_ERRO_ANALISE_EXPLORATORIA,"A matriz de contagens de grupos possui formato inválido","AnalisadorExploratorio.__validarResultados")
                    return False

                elif(np.asarray(dicionarioResultados[CHAVE_MAGNITUDE_MEDIA_DB]).shape != (QUANTIDADE_AMBIENTES,QUANTIDADE_PONTOS_FREQUENCIA)):
                    AnalisadorExploratorio.__registrarErro(CODIGO_ERRO_ANALISE_EXPLORATORIA,"A matriz de magnitude média possui formato inválido","AnalisadorExploratorio.__validarResultados")
                    return False

                elif(np.asarray(dicionarioResultados[CHAVE_TEMPORAL_MEDIA_DB]).shape != (QUANTIDADE_AMBIENTES,QUANTIDADE_PONTOS_FREQUENCIA)):
                    AnalisadorExploratorio.__registrarErro(CODIGO_ERRO_ANALISE_EXPLORATORIA,"A matriz temporal média possui formato inválido","AnalisadorExploratorio.__validarResultados")
                    return False

                elif(np.asarray(dicionarioResultados[CHAVE_COORDENADAS_PCA]).ndim != 2):
                    AnalisadorExploratorio.__registrarErro(CODIGO_ERRO_ANALISE_EXPLORATORIA,"A matriz da PCA possui formato inválido","AnalisadorExploratorio.__validarResultados")
                    return False

                elif(np.asarray(dicionarioResultados[CHAVE_COORDENADAS_PCA]).shape[1] != 2):
                    AnalisadorExploratorio.__registrarErro(CODIGO_ERRO_ANALISE_EXPLORATORIA,"A PCA deve possuir duas componentes","AnalisadorExploratorio.__validarResultados")
                    return False

                elif(np.asarray(dicionarioResultados[CHAVE_VARIANCIA_PCA]).shape != (2,)):
                    AnalisadorExploratorio.__registrarErro(CODIGO_ERRO_ANALISE_EXPLORATORIA,"O vetor de variância da PCA possui formato inválido","AnalisadorExploratorio.__validarResultados")
                    return False

                else:
                    for chave in chavesObrigatorias:
                        valor = np.asarray(dicionarioResultados[chave])

                        if(not np.all(np.isfinite(valor))):
                            AnalisadorExploratorio.__registrarErro(CODIGO_ERRO_ANALISE_EXPLORATORIA,"Um resultado contém valores inválidos","AnalisadorExploratorio.__validarResultados",chave)
                            return False

                    return True

        except Exception as excecao:
            AnalisadorExploratorio.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado durante a validação dos resultados","AnalisadorExploratorio.__validarResultados",str(excecao))
            return False

    @staticmethod
    def salvarResultados(dicionarioResultados:dict[str,np.ndarray],diretorioResultados:Path | str=DIRETORIO_RESULTADOS,diretorioGraficos:Path | str=DIRETORIO_GRAFICOS) -> list[Path] | None:
        AnalisadorExploratorio.__limparUltimoErro()

        try:
            if(not AnalisadorExploratorio.__validarResultados(dicionarioResultados)):
                return None

            else:
                diretorioResultados = Path(diretorioResultados)
                diretorioGraficos = Path(diretorioGraficos)

                diretorioResultados.mkdir(parents=True,exist_ok=True)
                diretorioGraficos.mkdir(parents=True,exist_ok=True)

                frequenciasGhz = dicionarioResultados[CHAVE_FREQUENCIAS_GHZ]
                atrasosUs = dicionarioResultados[CHAVE_ATRASOS_US]
                contagensMedicoes = dicionarioResultados[CHAVE_CONTAGENS_MEDICOES]
                contagensGrupos = dicionarioResultados[CHAVE_CONTAGENS_GRUPOS]
                magnitudeMediaDb = dicionarioResultados[CHAVE_MAGNITUDE_MEDIA_DB]
                magnitudeDesvioDb = dicionarioResultados[CHAVE_MAGNITUDE_DESVIO_DB]
                temporalMediaDb = dicionarioResultados[CHAVE_TEMPORAL_MEDIA_DB]
                coordenadasPca = dicionarioResultados[CHAVE_COORDENADAS_PCA]
                classesPca = dicionarioResultados[CHAVE_CLASSES_PCA]
                varianciaPca = dicionarioResultados[CHAVE_VARIANCIA_PCA]
                errosRepeticoes = dicionarioResultados[CHAVE_ERROS_REPETICOES]
                classesEstabilidade = dicionarioResultados[CHAVE_CLASSES_ESTABILIDADE]
                resumoEstabilidade = dicionarioResultados[CHAVE_RESUMO_ESTABILIDADE]

                listaDeArquivos = []

                caminhoResumoGeral = diretorioResultados / "resumo_geral.csv"

                with open(caminhoResumoGeral,"w",encoding="utf-8",newline="") as arquivoCsv:
                    escritor = csv.writer(arquivoCsv)
                    escritor.writerow(["metrica","valor"])
                    escritor.writerow(["quantidade_medicoes",int(np.sum(contagensMedicoes))])
                    escritor.writerow(["quantidade_grupos",int(np.sum(contagensGrupos))])
                    escritor.writerow(["quantidade_classes",QUANTIDADE_AMBIENTES])
                    escritor.writerow(["quantidade_pontos_frequencia",frequenciasGhz.size])
                    escritor.writerow(["frequencia_inicial_ghz","%.9f" % frequenciasGhz[0]])
                    escritor.writerow(["frequencia_final_ghz","%.9f" % frequenciasGhz[-1]])
                    escritor.writerow(["medicoes_treino",int(np.sum(contagensMedicoes[0]))])
                    escritor.writerow(["medicoes_validacao",int(np.sum(contagensMedicoes[1]))])
                    escritor.writerow(["medicoes_teste",int(np.sum(contagensMedicoes[2]))])
                    escritor.writerow(["grupos_treino",int(np.sum(contagensGrupos[0]))])
                    escritor.writerow(["grupos_validacao",int(np.sum(contagensGrupos[1]))])
                    escritor.writerow(["grupos_teste",int(np.sum(contagensGrupos[2]))])
                    escritor.writerow(["variancia_pca_1","%.12f" % varianciaPca[0]])
                    escritor.writerow(["variancia_pca_2","%.12f" % varianciaPca[1]])

                listaDeArquivos.append(caminhoResumoGeral)

                caminhoResumoDivisao = diretorioResultados / "resumo_divisao.csv"
                nomesSubconjuntos = ["treino","validacao","teste"]

                with open(caminhoResumoDivisao,"w",encoding="utf-8",newline="") as arquivoCsv:
                    escritor = csv.writer(arquivoCsv)
                    escritor.writerow(["subconjunto","classe","ambiente","quantidade_medicoes","quantidade_grupos"])

                    for indiceSubconjunto in range(3):
                        for classe in range(QUANTIDADE_AMBIENTES):
                            escritor.writerow([
                                nomesSubconjuntos[indiceSubconjunto],
                                classe,
                                DICIONARIO_NOMES_POR_CLASSE[classe],
                                int(contagensMedicoes[indiceSubconjunto,classe]),
                                int(contagensGrupos[indiceSubconjunto,classe])
                            ])

                listaDeArquivos.append(caminhoResumoDivisao)

                caminhoMagnitude = diretorioResultados / "magnitude_media_treino.csv"

                with open(caminhoMagnitude,"w",encoding="utf-8",newline="") as arquivoCsv:
                    escritor = csv.writer(arquivoCsv)
                    cabecalho = ["frequencia_ghz"]

                    for classe in range(QUANTIDADE_AMBIENTES):
                        cabecalho.append("media_db_classe_" + str(classe))
                        cabecalho.append("desvio_db_classe_" + str(classe))

                    escritor.writerow(cabecalho)

                    for indiceFrequencia in range(frequenciasGhz.size):
                        linha = ["%.12f" % frequenciasGhz[indiceFrequencia]]

                        for classe in range(QUANTIDADE_AMBIENTES):
                            linha.append("%.12f" % magnitudeMediaDb[classe,indiceFrequencia])
                            linha.append("%.12f" % magnitudeDesvioDb[classe,indiceFrequencia])

                        escritor.writerow(linha)

                listaDeArquivos.append(caminhoMagnitude)

                caminhoTemporal = diretorioResultados / "resposta_temporal_media_treino.csv"

                with open(caminhoTemporal,"w",encoding="utf-8",newline="") as arquivoCsv:
                    escritor = csv.writer(arquivoCsv)
                    cabecalho = ["atraso_us"]

                    for classe in range(QUANTIDADE_AMBIENTES):
                        cabecalho.append("magnitude_normalizada_db_classe_" + str(classe))

                    escritor.writerow(cabecalho)

                    for indiceAtraso in range(atrasosUs.size):
                        linha = ["%.12f" % atrasosUs[indiceAtraso]]

                        for classe in range(QUANTIDADE_AMBIENTES):
                            linha.append("%.12f" % temporalMediaDb[classe,indiceAtraso])

                        escritor.writerow(linha)

                listaDeArquivos.append(caminhoTemporal)

                caminhoEstabilidade = diretorioResultados / "resumo_estabilidade_repeticoes.csv"

                with open(caminhoEstabilidade,"w",encoding="utf-8",newline="") as arquivoCsv:
                    escritor = csv.writer(arquivoCsv)
                    escritor.writerow(["classe","ambiente","media_rmse_db","desvio_rmse_db","mediana_rmse_db","percentil_95_rmse_db","quantidade_medicoes"])

                    for classe in range(QUANTIDADE_AMBIENTES):
                        escritor.writerow([
                            classe,
                            DICIONARIO_NOMES_POR_CLASSE[classe],
                            "%.12f" % resumoEstabilidade[classe,0],
                            "%.12f" % resumoEstabilidade[classe,1],
                            "%.12f" % resumoEstabilidade[classe,2],
                            "%.12f" % resumoEstabilidade[classe,3],
                            int(resumoEstabilidade[classe,4])
                        ])

                listaDeArquivos.append(caminhoEstabilidade)

                caminhoPca = diretorioResultados / "pca_treino_cartesiana.csv"

                with open(caminhoPca,"w",encoding="utf-8",newline="") as arquivoCsv:
                    escritor = csv.writer(arquivoCsv)
                    escritor.writerow(["componente_1","componente_2","classe","ambiente"])

                    for indiceMedicao in range(coordenadasPca.shape[0]):
                        classe = int(classesPca[indiceMedicao])

                        escritor.writerow([
                            "%.12f" % coordenadasPca[indiceMedicao,0],
                            "%.12f" % coordenadasPca[indiceMedicao,1],
                            classe,
                            DICIONARIO_NOMES_POR_CLASSE[classe]
                        ])

                listaDeArquivos.append(caminhoPca)

                caminhoGraficoDivisao = diretorioGraficos / "distribuicao_divisao.png"
                posicoesClasses = np.arange(QUANTIDADE_AMBIENTES)
                larguraBarra = 0.25

                plt.figure(figsize=(10,6))

                for indiceSubconjunto in range(3):
                    deslocamento = (indiceSubconjunto - 1) * larguraBarra
                    plt.bar(posicoesClasses + deslocamento,contagensMedicoes[indiceSubconjunto],width=larguraBarra,label=nomesSubconjuntos[indiceSubconjunto].capitalize())

                listaDeNomesClasses = []

                for classe in range(QUANTIDADE_AMBIENTES):
                    listaDeNomesClasses.append(DICIONARIO_NOMES_POR_CLASSE[classe])

                plt.xticks(posicoesClasses,listaDeNomesClasses)
                plt.xlabel("Ambiente")
                plt.ylabel("Quantidade de medições")
                plt.title("Distribuição das medições por subconjunto")
                plt.legend()
                plt.tight_layout()
                plt.savefig(caminhoGraficoDivisao,dpi=300,bbox_inches="tight")
                plt.close()

                listaDeArquivos.append(caminhoGraficoDivisao)

                caminhoGraficoMagnitude = diretorioGraficos / "magnitude_media_treino.png"

                plt.figure(figsize=(10,6))

                for classe in range(QUANTIDADE_AMBIENTES):
                    plt.plot(frequenciasGhz,magnitudeMediaDb[classe],label=DICIONARIO_NOMES_POR_CLASSE[classe])

                plt.xlabel("Frequência (GHz)")
                plt.ylabel("Magnitude média de S21 (dB)")
                plt.title("Resposta média em frequência no subconjunto de treino")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(caminhoGraficoMagnitude,dpi=300,bbox_inches="tight")
                plt.close()

                listaDeArquivos.append(caminhoGraficoMagnitude)

                caminhoGraficoTemporal = diretorioGraficos / "resposta_temporal_media_treino.png"

                plt.figure(figsize=(10,6))

                for classe in range(QUANTIDADE_AMBIENTES):
                    plt.plot(atrasosUs,temporalMediaDb[classe],label=DICIONARIO_NOMES_POR_CLASSE[classe])

                plt.xlabel("Atraso (µs)")
                plt.ylabel("Magnitude média normalizada (dB)")
                plt.title("Resposta temporal média no subconjunto de treino")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(caminhoGraficoTemporal,dpi=300,bbox_inches="tight")
                plt.close()

                listaDeArquivos.append(caminhoGraficoTemporal)

                caminhoGraficoPca = diretorioGraficos / "pca_treino_cartesiana.png"

                plt.figure(figsize=(9,7))

                for classe in range(QUANTIDADE_AMBIENTES):
                    mascaraClasse = classesPca == classe
                    plt.scatter(coordenadasPca[mascaraClasse,0],coordenadasPca[mascaraClasse,1],s=10,alpha=0.55,label=DICIONARIO_NOMES_POR_CLASSE[classe])

                percentualComponente1 = 100.0 * varianciaPca[0]
                percentualComponente2 = 100.0 * varianciaPca[1]

                plt.xlabel("Componente principal 1 (%.2f%%)" % percentualComponente1)
                plt.ylabel("Componente principal 2 (%.2f%%)" % percentualComponente2)
                plt.title("PCA da representação cartesiana normalizada de treino")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(caminhoGraficoPca,dpi=300,bbox_inches="tight")
                plt.close()

                listaDeArquivos.append(caminhoGraficoPca)

                caminhoGraficoEstabilidade = diretorioGraficos / "estabilidade_repeticoes_treino.png"
                listaDeErrosPorClasse = []

                for classe in range(QUANTIDADE_AMBIENTES):
                    listaDeErrosPorClasse.append(errosRepeticoes[classesEstabilidade == classe])

                plt.figure(figsize=(10,6))
                plt.boxplot(listaDeErrosPorClasse,showfliers=False)
                plt.xticks(np.arange(1,QUANTIDADE_AMBIENTES + 1),listaDeNomesClasses)
                plt.xlabel("Ambiente")
                plt.ylabel("RMSE em relação à média da posição (dB)")
                plt.title("Estabilidade das medições repetidas no subconjunto de treino")
                plt.grid(True,axis="y")
                plt.tight_layout()
                plt.savefig(caminhoGraficoEstabilidade,dpi=300,bbox_inches="tight")
                plt.close()

                listaDeArquivos.append(caminhoGraficoEstabilidade)

                caminhoGraficoDesvioMagnitude = diretorioGraficos / "desvio_magnitude_treino.png"

                plt.figure(figsize=(10,6))

                for classe in range(QUANTIDADE_AMBIENTES):
                    plt.plot(frequenciasGhz,magnitudeDesvioDb[classe],label=DICIONARIO_NOMES_POR_CLASSE[classe])

                plt.xlabel("Frequência (GHz)")
                plt.ylabel("Desvio padrão da magnitude (dB)")
                plt.title("Variabilidade da resposta em frequência no subconjunto de treino")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(caminhoGraficoDesvioMagnitude,dpi=300,bbox_inches="tight")
                plt.close()

                listaDeArquivos.append(caminhoGraficoDesvioMagnitude)

                return listaDeArquivos

        except Exception as excecao:
            AnalisadorExploratorio.__registrarErro(CODIGO_ERRO_SALVAMENTO_ANALISE,"Não foi possível salvar os resultados da análise exploratória","AnalisadorExploratorio.salvarResultados",str(excecao))
            return None

    @staticmethod
    def executarAnalise() -> list[Path] | None:
        AnalisadorExploratorio.__limparUltimoErro()

        try:
            dicionarioBase = ProcessadorBase.carregarBase(CAMINHO_BASE_PROCESSADA,True)

            if(dicionarioBase is None):
                AnalisadorExploratorio.__copiarErro(ProcessadorBase.getUltimoErro(),CODIGO_ERRO_ANALISE_EXPLORATORIA,"Não foi possível carregar a base processada","AnalisadorExploratorio.executarAnalise")
                return None

            else:
                dicionarioBasePreparada = PreparadorExperimentos.carregarBasePreparada(CAMINHO_BASE_PREPARADA)

                if(dicionarioBasePreparada is None):
                    AnalisadorExploratorio.__copiarErro(PreparadorExperimentos.getUltimoErro(),CODIGO_ERRO_ANALISE_EXPLORATORIA,"Não foi possível carregar a base preparada","AnalisadorExploratorio.executarAnalise")
                    return None

                else:
                    dicionarioResultados = AnalisadorExploratorio.gerarResultados(dicionarioBase,dicionarioBasePreparada)

                    if(dicionarioResultados is None):
                        return None

                    else:
                        return AnalisadorExploratorio.salvarResultados(dicionarioResultados)

        except Exception as excecao:
            AnalisadorExploratorio.__registrarErro(CODIGO_ERRO_ANALISE_EXPLORATORIA,"Não foi possível executar a análise exploratória","AnalisadorExploratorio.executarAnalise",str(excecao))
            return None