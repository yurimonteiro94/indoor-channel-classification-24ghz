from __future__ import annotations

from pathlib import Path

import numpy as np

from model.entidades.erro_sistema import ErroSistema
from services.constantes import CAMINHO_BASE_PREPARADA
from services.constantes import CAMINHO_BASE_PROCESSADA
from services.constantes import CHAVE_AMBIENTES
from services.constantes import CHAVE_ATRASOS
from services.constantes import CHAVE_CLASSES
from services.constantes import CHAVE_DADOS_NORMALIZADOS
from services.constantes import CHAVE_DESVIO_CARTESIANA
from services.constantes import CHAVE_DESVIO_POLAR
from services.constantes import CHAVE_DESVIO_TEMPORAL
from services.constantes import CHAVE_DESVIOS
from services.constantes import CHAVE_FREQUENCIAS
from services.constantes import CHAVE_GRUPOS
from services.constantes import CHAVE_GRUPOS_TESTE
from services.constantes import CHAVE_GRUPOS_TREINO
from services.constantes import CHAVE_GRUPOS_VALIDACAO
from services.constantes import CHAVE_INDICES_TESTE
from services.constantes import CHAVE_INDICES_TREINO
from services.constantes import CHAVE_INDICES_VALIDACAO
from services.constantes import CHAVE_MEDIA_CARTESIANA
from services.constantes import CHAVE_MEDIA_POLAR
from services.constantes import CHAVE_MEDIA_TEMPORAL
from services.constantes import CHAVE_MEDIAS
from services.constantes import CHAVE_POSICOES
from services.constantes import CHAVE_REPETICOES
from services.constantes import CHAVE_REPRESENTACAO_CARTESIANA
from services.constantes import CHAVE_REPRESENTACAO_POLAR
from services.constantes import CHAVE_REPRESENTACAO_TEMPORAL
from services.constantes import CHAVE_S21
from services.constantes import CODIGO_ERRO_BASE_PREPARADA_INVALIDA
from services.constantes import CODIGO_ERRO_CARREGAMENTO_BASE_PREPARADA
from services.constantes import CODIGO_ERRO_DESCONHECIDO
from services.constantes import CODIGO_ERRO_PREPARACAO_EXPERIMENTOS
from services.constantes import CODIGO_ERRO_SALVAMENTO_BASE_PREPARADA
from services.constantes import QUANTIDADE_CANAIS_S21
from services.constantes import QUANTIDADE_PONTOS_FREQUENCIA
from services.gerador_representacoes import GeradorRepresentacoes
from services.normalizador_dados import NormalizadorDados
from services.processador_base import ProcessadorBase
from services.separador_base import SeparadorBase


class PreparadorExperimentos:
    __ultimoErro = None

    @staticmethod
    def __registrarErro(codigo:int,mensagem:str,origem:str,detalhe:str | None=None) -> None:
        PreparadorExperimentos.__ultimoErro = ErroSistema.criar(codigo,mensagem,origem,detalhe)

    @staticmethod
    def __limparUltimoErro() -> None:
        PreparadorExperimentos.__ultimoErro = None

    @staticmethod
    def __copiarErro(erro:ErroSistema | None,codigo:int,mensagem:str,origem:str) -> None:
        if(erro is None):
            PreparadorExperimentos.__registrarErro(codigo,mensagem,origem)

        else:
            PreparadorExperimentos.__ultimoErro = erro

    @staticmethod
    def getUltimoErro() -> ErroSistema | None:
        return PreparadorExperimentos.__ultimoErro

    @staticmethod
    def ultimaExecucaoDeuErro() -> bool:
        if(PreparadorExperimentos.__ultimoErro is None):
            return False

        else:
            return True

    @staticmethod
    def prepararBase(dicionarioBase:dict[str,np.ndarray],validarBaseCompleta:bool=False) -> dict[str,np.ndarray] | None:
        PreparadorExperimentos.__limparUltimoErro()

        try:
            if(not ProcessadorBase.validarBase(dicionarioBase,validarBaseCompleta)):
                PreparadorExperimentos.__copiarErro(ProcessadorBase.getUltimoErro(),CODIGO_ERRO_PREPARACAO_EXPERIMENTOS,"A base original é inválida","PreparadorExperimentos.prepararBase")
                return None

            else:
                matrizS21 = np.asarray(dicionarioBase[CHAVE_S21])
                vetorClasses = np.asarray(dicionarioBase[CHAVE_CLASSES])
                vetorGrupos = np.asarray(dicionarioBase[CHAVE_GRUPOS])

                dicionarioDivisao = SeparadorBase.criarDivisao(vetorClasses,vetorGrupos)

                if(dicionarioDivisao is None):
                    PreparadorExperimentos.__copiarErro(SeparadorBase.getUltimoErro(),CODIGO_ERRO_PREPARACAO_EXPERIMENTOS,"Não foi possível dividir a base","PreparadorExperimentos.prepararBase")
                    return None

                else:
                    indicesTreino = dicionarioDivisao[CHAVE_INDICES_TREINO]

                    representacaoCartesiana = GeradorRepresentacoes.gerarCartesiana(matrizS21)

                    if(representacaoCartesiana is None):
                        PreparadorExperimentos.__copiarErro(GeradorRepresentacoes.getUltimoErro(),CODIGO_ERRO_PREPARACAO_EXPERIMENTOS,"Não foi possível gerar a representação cartesiana","PreparadorExperimentos.prepararBase")
                        return None

                    else:
                        resultadoCartesiano = NormalizadorDados.normalizarPorCanais(representacaoCartesiana,indicesTreino)
                        del representacaoCartesiana

                        if(resultadoCartesiano is None):
                            PreparadorExperimentos.__copiarErro(NormalizadorDados.getUltimoErro(),CODIGO_ERRO_PREPARACAO_EXPERIMENTOS,"Não foi possível normalizar a representação cartesiana","PreparadorExperimentos.prepararBase")
                            return None

                        else:
                            representacaoPolar = GeradorRepresentacoes.gerarPolar(matrizS21)

                            if(representacaoPolar is None):
                                PreparadorExperimentos.__copiarErro(GeradorRepresentacoes.getUltimoErro(),CODIGO_ERRO_PREPARACAO_EXPERIMENTOS,"Não foi possível gerar a representação polar","PreparadorExperimentos.prepararBase")
                                return None

                            else:
                                resultadoPolar = NormalizadorDados.normalizarPorCanais(representacaoPolar,indicesTreino)
                                del representacaoPolar

                                if(resultadoPolar is None):
                                    PreparadorExperimentos.__copiarErro(NormalizadorDados.getUltimoErro(),CODIGO_ERRO_PREPARACAO_EXPERIMENTOS,"Não foi possível normalizar a representação polar","PreparadorExperimentos.prepararBase")
                                    return None

                                else:
                                    representacaoTemporal = GeradorRepresentacoes.gerarTemporal(matrizS21)

                                    if(representacaoTemporal is None):
                                        PreparadorExperimentos.__copiarErro(GeradorRepresentacoes.getUltimoErro(),CODIGO_ERRO_PREPARACAO_EXPERIMENTOS,"Não foi possível gerar a representação temporal","PreparadorExperimentos.prepararBase")
                                        return None

                                    else:
                                        resultadoTemporal = NormalizadorDados.normalizarPorCanais(representacaoTemporal,indicesTreino)
                                        del representacaoTemporal

                                        if(resultadoTemporal is None):
                                            PreparadorExperimentos.__copiarErro(NormalizadorDados.getUltimoErro(),CODIGO_ERRO_PREPARACAO_EXPERIMENTOS,"Não foi possível normalizar a representação temporal","PreparadorExperimentos.prepararBase")
                                            return None

                                        else:
                                            vetorAtrasos = GeradorRepresentacoes.obterVetorAtrasos()

                                            if(vetorAtrasos is None):
                                                PreparadorExperimentos.__copiarErro(GeradorRepresentacoes.getUltimoErro(),CODIGO_ERRO_PREPARACAO_EXPERIMENTOS,"Não foi possível gerar o vetor de atrasos","PreparadorExperimentos.prepararBase")
                                                return None

                                            else:
                                                dicionarioBasePreparada = {
                                                    CHAVE_FREQUENCIAS: np.asarray(dicionarioBase[CHAVE_FREQUENCIAS]).copy(),
                                                    CHAVE_ATRASOS: vetorAtrasos,
                                                    CHAVE_CLASSES: np.asarray(dicionarioBase[CHAVE_CLASSES]).copy(),
                                                    CHAVE_AMBIENTES: np.asarray(dicionarioBase[CHAVE_AMBIENTES]).copy(),
                                                    CHAVE_POSICOES: np.asarray(dicionarioBase[CHAVE_POSICOES]).copy(),
                                                    CHAVE_REPETICOES: np.asarray(dicionarioBase[CHAVE_REPETICOES]).copy(),
                                                    CHAVE_GRUPOS: np.asarray(dicionarioBase[CHAVE_GRUPOS]).copy(),
                                                    CHAVE_INDICES_TREINO: dicionarioDivisao[CHAVE_INDICES_TREINO],
                                                    CHAVE_INDICES_VALIDACAO: dicionarioDivisao[CHAVE_INDICES_VALIDACAO],
                                                    CHAVE_INDICES_TESTE: dicionarioDivisao[CHAVE_INDICES_TESTE],
                                                    CHAVE_GRUPOS_TREINO: dicionarioDivisao[CHAVE_GRUPOS_TREINO],
                                                    CHAVE_GRUPOS_VALIDACAO: dicionarioDivisao[CHAVE_GRUPOS_VALIDACAO],
                                                    CHAVE_GRUPOS_TESTE: dicionarioDivisao[CHAVE_GRUPOS_TESTE],
                                                    CHAVE_REPRESENTACAO_CARTESIANA: resultadoCartesiano[CHAVE_DADOS_NORMALIZADOS],
                                                    CHAVE_MEDIA_CARTESIANA: resultadoCartesiano[CHAVE_MEDIAS],
                                                    CHAVE_DESVIO_CARTESIANA: resultadoCartesiano[CHAVE_DESVIOS],
                                                    CHAVE_REPRESENTACAO_POLAR: resultadoPolar[CHAVE_DADOS_NORMALIZADOS],
                                                    CHAVE_MEDIA_POLAR: resultadoPolar[CHAVE_MEDIAS],
                                                    CHAVE_DESVIO_POLAR: resultadoPolar[CHAVE_DESVIOS],
                                                    CHAVE_REPRESENTACAO_TEMPORAL: resultadoTemporal[CHAVE_DADOS_NORMALIZADOS],
                                                    CHAVE_MEDIA_TEMPORAL: resultadoTemporal[CHAVE_MEDIAS],
                                                    CHAVE_DESVIO_TEMPORAL: resultadoTemporal[CHAVE_DESVIOS]
                                                }

                                                if(not PreparadorExperimentos.validarBasePreparada(dicionarioBasePreparada)):
                                                    return None

                                                else:
                                                    return dicionarioBasePreparada

        except Exception as excecao:
            PreparadorExperimentos.__registrarErro(CODIGO_ERRO_PREPARACAO_EXPERIMENTOS,"Erro inesperado durante a preparação dos experimentos","PreparadorExperimentos.prepararBase",str(excecao))
            return None

    @staticmethod
    def validarBasePreparada(dicionarioBasePreparada:dict[str,np.ndarray]) -> bool:
        PreparadorExperimentos.__limparUltimoErro()

        try:
            if(not isinstance(dicionarioBasePreparada,dict)):
                PreparadorExperimentos.__registrarErro(CODIGO_ERRO_BASE_PREPARADA_INVALIDA,"A base preparada não é um dicionário","PreparadorExperimentos.validarBasePreparada")
                return False

            else:
                chavesObrigatorias = (
                    CHAVE_FREQUENCIAS,CHAVE_ATRASOS,CHAVE_CLASSES,CHAVE_AMBIENTES,CHAVE_POSICOES,CHAVE_REPETICOES,CHAVE_GRUPOS,
                    CHAVE_INDICES_TREINO,CHAVE_INDICES_VALIDACAO,CHAVE_INDICES_TESTE,
                    CHAVE_GRUPOS_TREINO,CHAVE_GRUPOS_VALIDACAO,CHAVE_GRUPOS_TESTE,
                    CHAVE_REPRESENTACAO_CARTESIANA,CHAVE_MEDIA_CARTESIANA,CHAVE_DESVIO_CARTESIANA,
                    CHAVE_REPRESENTACAO_POLAR,CHAVE_MEDIA_POLAR,CHAVE_DESVIO_POLAR,
                    CHAVE_REPRESENTACAO_TEMPORAL,CHAVE_MEDIA_TEMPORAL,CHAVE_DESVIO_TEMPORAL
                )

                for chave in chavesObrigatorias:
                    if(chave not in dicionarioBasePreparada):
                        PreparadorExperimentos.__registrarErro(CODIGO_ERRO_BASE_PREPARADA_INVALIDA,"Uma chave obrigatória não foi encontrada","PreparadorExperimentos.validarBasePreparada",chave)
                        return False

                vetorClasses = np.asarray(dicionarioBasePreparada[CHAVE_CLASSES])
                vetorGrupos = np.asarray(dicionarioBasePreparada[CHAVE_GRUPOS])
                vetorFrequencias = np.asarray(dicionarioBasePreparada[CHAVE_FREQUENCIAS])
                vetorAtrasos = np.asarray(dicionarioBasePreparada[CHAVE_ATRASOS])
                quantidadeDeMedicoes = vetorClasses.size

                if(vetorClasses.ndim != 1 or vetorGrupos.shape != (quantidadeDeMedicoes,)):
                    PreparadorExperimentos.__registrarErro(CODIGO_ERRO_BASE_PREPARADA_INVALIDA,"Os vetores de classes ou grupos são inválidos","PreparadorExperimentos.validarBasePreparada")
                    return False

                elif(np.asarray(dicionarioBasePreparada[CHAVE_AMBIENTES]).shape != (quantidadeDeMedicoes,)):
                    PreparadorExperimentos.__registrarErro(CODIGO_ERRO_BASE_PREPARADA_INVALIDA,"O vetor de ambientes é inválido","PreparadorExperimentos.validarBasePreparada")
                    return False

                elif(np.asarray(dicionarioBasePreparada[CHAVE_POSICOES]).shape != (quantidadeDeMedicoes,)):
                    PreparadorExperimentos.__registrarErro(CODIGO_ERRO_BASE_PREPARADA_INVALIDA,"O vetor de posições é inválido","PreparadorExperimentos.validarBasePreparada")
                    return False

                elif(np.asarray(dicionarioBasePreparada[CHAVE_REPETICOES]).shape != (quantidadeDeMedicoes,)):
                    PreparadorExperimentos.__registrarErro(CODIGO_ERRO_BASE_PREPARADA_INVALIDA,"O vetor de repetições é inválido","PreparadorExperimentos.validarBasePreparada")
                    return False

                elif(vetorFrequencias.shape != (QUANTIDADE_PONTOS_FREQUENCIA,)):
                    PreparadorExperimentos.__registrarErro(CODIGO_ERRO_BASE_PREPARADA_INVALIDA,"O vetor de frequências é inválido","PreparadorExperimentos.validarBasePreparada")
                    return False

                elif(vetorAtrasos.shape != (QUANTIDADE_PONTOS_FREQUENCIA,)):
                    PreparadorExperimentos.__registrarErro(CODIGO_ERRO_BASE_PREPARADA_INVALIDA,"O vetor de atrasos é inválido","PreparadorExperimentos.validarBasePreparada")
                    return False

                elif(not np.all(np.isfinite(vetorFrequencias)) or not np.all(np.isfinite(vetorAtrasos))):
                    PreparadorExperimentos.__registrarErro(CODIGO_ERRO_BASE_PREPARADA_INVALIDA,"Os vetores de frequências ou atrasos contêm valores inválidos","PreparadorExperimentos.validarBasePreparada")
                    return False

                elif(not np.all(np.diff(vetorAtrasos) > 0.0)):
                    PreparadorExperimentos.__registrarErro(CODIGO_ERRO_BASE_PREPARADA_INVALIDA,"O vetor de atrasos não é crescente","PreparadorExperimentos.validarBasePreparada")
                    return False

                else:
                    listaDeRepresentacoes = [
                        ("cartesiana",np.asarray(dicionarioBasePreparada[CHAVE_REPRESENTACAO_CARTESIANA])),
                        ("polar",np.asarray(dicionarioBasePreparada[CHAVE_REPRESENTACAO_POLAR])),
                        ("temporal",np.asarray(dicionarioBasePreparada[CHAVE_REPRESENTACAO_TEMPORAL]))
                    ]

                    for item in listaDeRepresentacoes:
                        nomeRepresentacao = item[0]
                        matrizRepresentacao = item[1]

                        if(matrizRepresentacao.shape != (quantidadeDeMedicoes,QUANTIDADE_PONTOS_FREQUENCIA,QUANTIDADE_CANAIS_S21)):
                            PreparadorExperimentos.__registrarErro(CODIGO_ERRO_BASE_PREPARADA_INVALIDA,"O formato de uma representação é inválido","PreparadorExperimentos.validarBasePreparada",nomeRepresentacao)
                            return False

                        elif(not np.all(np.isfinite(matrizRepresentacao))):
                            PreparadorExperimentos.__registrarErro(CODIGO_ERRO_BASE_PREPARADA_INVALIDA,"Uma representação contém valores inválidos","PreparadorExperimentos.validarBasePreparada",nomeRepresentacao)
                            return False

                    listaDeParametros = [
                        ("média cartesiana",np.asarray(dicionarioBasePreparada[CHAVE_MEDIA_CARTESIANA])),
                        ("desvio cartesiano",np.asarray(dicionarioBasePreparada[CHAVE_DESVIO_CARTESIANA])),
                        ("média polar",np.asarray(dicionarioBasePreparada[CHAVE_MEDIA_POLAR])),
                        ("desvio polar",np.asarray(dicionarioBasePreparada[CHAVE_DESVIO_POLAR])),
                        ("média temporal",np.asarray(dicionarioBasePreparada[CHAVE_MEDIA_TEMPORAL])),
                        ("desvio temporal",np.asarray(dicionarioBasePreparada[CHAVE_DESVIO_TEMPORAL]))
                    ]

                    for item in listaDeParametros:
                        nomeParametro = item[0]
                        vetorParametro = item[1]

                        if(vetorParametro.shape != (QUANTIDADE_CANAIS_S21,)):
                            PreparadorExperimentos.__registrarErro(CODIGO_ERRO_BASE_PREPARADA_INVALIDA,"O formato de um parâmetro é inválido","PreparadorExperimentos.validarBasePreparada",nomeParametro)
                            return False

                        elif(not np.all(np.isfinite(vetorParametro))):
                            PreparadorExperimentos.__registrarErro(CODIGO_ERRO_BASE_PREPARADA_INVALIDA,"Um parâmetro contém valores inválidos","PreparadorExperimentos.validarBasePreparada",nomeParametro)
                            return False

                    dicionarioDivisao = {
                        CHAVE_INDICES_TREINO: dicionarioBasePreparada[CHAVE_INDICES_TREINO],
                        CHAVE_INDICES_VALIDACAO: dicionarioBasePreparada[CHAVE_INDICES_VALIDACAO],
                        CHAVE_INDICES_TESTE: dicionarioBasePreparada[CHAVE_INDICES_TESTE],
                        CHAVE_GRUPOS_TREINO: dicionarioBasePreparada[CHAVE_GRUPOS_TREINO],
                        CHAVE_GRUPOS_VALIDACAO: dicionarioBasePreparada[CHAVE_GRUPOS_VALIDACAO],
                        CHAVE_GRUPOS_TESTE: dicionarioBasePreparada[CHAVE_GRUPOS_TESTE]
                    }

                    if(not SeparadorBase.validarDivisao(dicionarioDivisao,vetorClasses,vetorGrupos)):
                        PreparadorExperimentos.__copiarErro(SeparadorBase.getUltimoErro(),CODIGO_ERRO_BASE_PREPARADA_INVALIDA,"A divisão da base preparada é inválida","PreparadorExperimentos.validarBasePreparada")
                        return False

                    else:
                        return True

        except Exception as excecao:
            PreparadorExperimentos.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado durante a validação da base preparada","PreparadorExperimentos.validarBasePreparada",str(excecao))
            return False

    @staticmethod
    def salvarBasePreparada(dicionarioBasePreparada:dict[str,np.ndarray],caminhoBasePreparada:Path | str=CAMINHO_BASE_PREPARADA) -> bool:
        PreparadorExperimentos.__limparUltimoErro()

        try:
            if(not PreparadorExperimentos.validarBasePreparada(dicionarioBasePreparada)):
                return False

            else:
                caminhoBasePreparada = Path(caminhoBasePreparada)
                caminhoBasePreparada.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(caminhoBasePreparada,**dicionarioBasePreparada)
                return True

        except Exception as excecao:
            PreparadorExperimentos.__registrarErro(CODIGO_ERRO_SALVAMENTO_BASE_PREPARADA,"Não foi possível salvar a base preparada","PreparadorExperimentos.salvarBasePreparada",str(excecao))
            return False

    @staticmethod
    def carregarBasePreparada(caminhoBasePreparada:Path | str=CAMINHO_BASE_PREPARADA) -> dict[str,np.ndarray] | None:
        PreparadorExperimentos.__limparUltimoErro()

        try:
            caminhoBasePreparada = Path(caminhoBasePreparada)

            if(not caminhoBasePreparada.exists()):
                PreparadorExperimentos.__registrarErro(CODIGO_ERRO_CARREGAMENTO_BASE_PREPARADA,"A base preparada não foi encontrada","PreparadorExperimentos.carregarBasePreparada",str(caminhoBasePreparada))
                return None

            elif(not caminhoBasePreparada.is_file()):
                PreparadorExperimentos.__registrarErro(CODIGO_ERRO_CARREGAMENTO_BASE_PREPARADA,"O caminho da base preparada não representa um arquivo","PreparadorExperimentos.carregarBasePreparada",str(caminhoBasePreparada))
                return None

            else:
                dicionarioBasePreparada = {}

                with np.load(caminhoBasePreparada,allow_pickle=False) as arquivoBase:
                    for chave in arquivoBase.files:
                        dicionarioBasePreparada[chave] = arquivoBase[chave]

                if(not PreparadorExperimentos.validarBasePreparada(dicionarioBasePreparada)):
                    return None

                else:
                    return dicionarioBasePreparada

        except Exception as excecao:
            PreparadorExperimentos.__registrarErro(CODIGO_ERRO_CARREGAMENTO_BASE_PREPARADA,"Não foi possível carregar a base preparada","PreparadorExperimentos.carregarBasePreparada",str(excecao))
            return None

    @staticmethod
    def prepararESalvar(caminhoBaseProcessada:Path | str=CAMINHO_BASE_PROCESSADA,caminhoBasePreparada:Path | str=CAMINHO_BASE_PREPARADA) -> bool:
        PreparadorExperimentos.__limparUltimoErro()

        try:
            dicionarioBase = ProcessadorBase.carregarBase(caminhoBaseProcessada,True)

            if(dicionarioBase is None):
                PreparadorExperimentos.__copiarErro(ProcessadorBase.getUltimoErro(),CODIGO_ERRO_PREPARACAO_EXPERIMENTOS,"Não foi possível carregar a base original","PreparadorExperimentos.prepararESalvar")
                return False

            else:
                dicionarioBasePreparada = PreparadorExperimentos.prepararBase(dicionarioBase,True)

                if(dicionarioBasePreparada is None):
                    return False

                elif(not PreparadorExperimentos.salvarBasePreparada(dicionarioBasePreparada,caminhoBasePreparada)):
                    return False

                else:
                    return True

        except Exception as excecao:
            PreparadorExperimentos.__registrarErro(CODIGO_ERRO_PREPARACAO_EXPERIMENTOS,"Não foi possível preparar e salvar os experimentos","PreparadorExperimentos.prepararESalvar",str(excecao))
            return False