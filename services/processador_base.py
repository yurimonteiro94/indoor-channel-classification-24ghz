from __future__ import annotations

from pathlib import Path

import numpy as np

from model.dao.medicao_canal_dao import MedicaoCanalDAO
from model.entidades.erro_sistema import ErroSistema
from model.entidades.medicao_canal import MedicaoCanal
from services.constantes import CAMINHO_ARQUIVO_ZIP
from services.constantes import CAMINHO_BASE_PROCESSADA
from services.constantes import CHAVE_AMBIENTES
from services.constantes import CHAVE_ARQUIVOS
from services.constantes import CHAVE_CLASSES
from services.constantes import CHAVE_FREQUENCIAS
from services.constantes import CHAVE_GRUPOS
from services.constantes import CHAVE_POSICOES
from services.constantes import CHAVE_REPETICOES
from services.constantes import CHAVE_S21
from services.constantes import CODIGO_ERRO_ARGUMENTO_INVALIDO
from services.constantes import CODIGO_ERRO_BASE_COMPLETA_INVALIDA
from services.constantes import CODIGO_ERRO_BASE_INVALIDA
from services.constantes import CODIGO_ERRO_CARREGAMENTO_BASE
from services.constantes import CODIGO_ERRO_CHAVE_AUSENTE
from services.constantes import CODIGO_ERRO_CONTEUDO_BASE_INVALIDO
from services.constantes import CODIGO_ERRO_DESCONHECIDO
from services.constantes import CODIGO_ERRO_FORMATO_BASE_INVALIDO
from services.constantes import CODIGO_ERRO_PROCESSAMENTO_BASE
from services.constantes import CODIGO_ERRO_SALVAMENTO_BASE
from services.constantes import FREQUENCIA_FINAL_HZ
from services.constantes import FREQUENCIA_INICIAL_HZ
from services.constantes import QUANTIDADE_AMBIENTES
from services.constantes import QUANTIDADE_CANAIS_S21
from services.constantes import QUANTIDADE_GRUPOS_TOTAL
from services.constantes import QUANTIDADE_MEDICOES_POR_AMBIENTE
from services.constantes import QUANTIDADE_MEDICOES_TOTAL
from services.constantes import QUANTIDADE_PONTOS_FREQUENCIA
from services.constantes import QUANTIDADE_REPETICOES_POR_POSICAO
from services.constantes import TOLERANCIA_FREQUENCIA_HZ


class ProcessadorBase:
    __ultimoErro = None

    @staticmethod
    def __registrarErro(codigo:int,mensagem:str,origem:str,detalhe:str | None=None) -> None:
        ProcessadorBase.__ultimoErro = ErroSistema.criar(codigo,mensagem,origem,detalhe)

    @staticmethod
    def __limparUltimoErro() -> None:
        ProcessadorBase.__ultimoErro = None

    @staticmethod
    def getUltimoErro() -> ErroSistema | None:
        return ProcessadorBase.__ultimoErro

    @staticmethod
    def ultimaExecucaoDeuErro() -> bool:
        if(ProcessadorBase.__ultimoErro is None):
            return False

        else:
            return True

    @staticmethod
    def construirBase(caminhoArquivoZip:Path | str=CAMINHO_ARQUIVO_ZIP,quantidadeMaxima:int | None=None) -> dict[str,np.ndarray] | None:
        ProcessadorBase.__limparUltimoErro()

        try:
            listaDeMedicoes = MedicaoCanalDAO.carregarMedicoes(caminhoArquivoZip,quantidadeMaxima)

            if(listaDeMedicoes is None):
                ProcessadorBase.__ultimoErro = MedicaoCanalDAO.getUltimoErro()
                return None

            elif(len(listaDeMedicoes) == 0):
                ProcessadorBase.__registrarErro(CODIGO_ERRO_PROCESSAMENTO_BASE,"Não existem medições para processar","ProcessadorBase.construirBase")
                return None

            else:
                quantidadeDeMedicoes = len(listaDeMedicoes)
                matrizS21 = np.empty((quantidadeDeMedicoes,QUANTIDADE_PONTOS_FREQUENCIA,QUANTIDADE_CANAIS_S21),dtype=np.float32)
                vetorClasses = np.empty(quantidadeDeMedicoes,dtype=np.int8)
                vetorRepeticoes = np.empty(quantidadeDeMedicoes,dtype=np.int8)
                vetorGrupos = np.empty(quantidadeDeMedicoes,dtype=np.int16)

                listaDeAmbientes = []
                listaDePosicoes = []
                listaDeArquivos = []
                vetorFrequencias = None

                for indice in range(quantidadeDeMedicoes):
                    medicao = listaDeMedicoes[indice]
                    matrizDaMedicao = medicao.obterS21EmDoisCanais()

                    if(matrizDaMedicao is None):
                        ProcessadorBase.__ultimoErro = MedicaoCanal.getUltimoErro()
                        return None

                    else:
                        matrizS21[indice] = matrizDaMedicao
                        vetorClasses[indice] = medicao.getClasse()
                        vetorRepeticoes[indice] = medicao.getRepeticao()
                        vetorGrupos[indice] = medicao.getGrupo()

                        listaDeAmbientes.append(medicao.getNomeAmbiente())
                        listaDePosicoes.append(medicao.getPosicao())
                        listaDeArquivos.append(medicao.getCaminhoArquivo())

                        if(vetorFrequencias is None):
                            vetorFrequencias = medicao.getFrequencias()

                dicionarioBase = {
                    CHAVE_FREQUENCIAS: vetorFrequencias,
                    CHAVE_S21: matrizS21,
                    CHAVE_CLASSES: vetorClasses,
                    CHAVE_AMBIENTES: np.asarray(listaDeAmbientes,dtype=np.str_),
                    CHAVE_POSICOES: np.asarray(listaDePosicoes,dtype=np.str_),
                    CHAVE_REPETICOES: vetorRepeticoes,
                    CHAVE_GRUPOS: vetorGrupos,
                    CHAVE_ARQUIVOS: np.asarray(listaDeArquivos,dtype=np.str_)
                }

                return dicionarioBase

        except Exception as excecao:
            ProcessadorBase.__registrarErro(CODIGO_ERRO_PROCESSAMENTO_BASE,"Erro inesperado durante o processamento da base","ProcessadorBase.construirBase",str(excecao))
            return None

    @staticmethod
    def validarBase(dicionarioBase:dict[str,np.ndarray],validarBaseCompleta:bool=False) -> bool:
        ProcessadorBase.__limparUltimoErro()

        try:
            if(not isinstance(dicionarioBase,dict)):
                ProcessadorBase.__registrarErro(CODIGO_ERRO_BASE_INVALIDA,"A base informada não é um dicionário","ProcessadorBase.validarBase")
                return False

            else:
                chavesObrigatorias = (CHAVE_FREQUENCIAS,CHAVE_S21,CHAVE_CLASSES,CHAVE_AMBIENTES,CHAVE_POSICOES,CHAVE_REPETICOES,CHAVE_GRUPOS,CHAVE_ARQUIVOS)

                for chave in chavesObrigatorias:
                    if(chave not in dicionarioBase):
                        ProcessadorBase.__registrarErro(CODIGO_ERRO_CHAVE_AUSENTE,"Uma chave obrigatória não foi encontrada na base","ProcessadorBase.validarBase",chave)
                        return False

                vetorFrequencias = np.asarray(dicionarioBase[CHAVE_FREQUENCIAS])
                matrizS21 = np.asarray(dicionarioBase[CHAVE_S21])
                vetorClasses = np.asarray(dicionarioBase[CHAVE_CLASSES])
                vetorAmbientes = np.asarray(dicionarioBase[CHAVE_AMBIENTES])
                vetorPosicoes = np.asarray(dicionarioBase[CHAVE_POSICOES])
                vetorRepeticoes = np.asarray(dicionarioBase[CHAVE_REPETICOES])
                vetorGrupos = np.asarray(dicionarioBase[CHAVE_GRUPOS])
                vetorArquivos = np.asarray(dicionarioBase[CHAVE_ARQUIVOS])

                if(vetorClasses.ndim != 1):
                    ProcessadorBase.__registrarErro(CODIGO_ERRO_FORMATO_BASE_INVALIDO,"O vetor de classes deve possuir uma dimensão","ProcessadorBase.validarBase")
                    return False

                else:
                    quantidadeDeMedicoes = vetorClasses.shape[0]

                    if(vetorFrequencias.shape != (QUANTIDADE_PONTOS_FREQUENCIA,)):
                        ProcessadorBase.__registrarErro(CODIGO_ERRO_FORMATO_BASE_INVALIDO,"O formato do vetor de frequências é inválido","ProcessadorBase.validarBase",str(vetorFrequencias.shape))
                        return False

                    elif(matrizS21.shape != (quantidadeDeMedicoes,QUANTIDADE_PONTOS_FREQUENCIA,QUANTIDADE_CANAIS_S21)):
                        ProcessadorBase.__registrarErro(CODIGO_ERRO_FORMATO_BASE_INVALIDO,"O formato da matriz S21 é inválido","ProcessadorBase.validarBase",str(matrizS21.shape))
                        return False

                    elif(vetorAmbientes.shape != (quantidadeDeMedicoes,)):
                        ProcessadorBase.__registrarErro(CODIGO_ERRO_FORMATO_BASE_INVALIDO,"O formato do vetor de ambientes é inválido","ProcessadorBase.validarBase",str(vetorAmbientes.shape))
                        return False

                    elif(vetorPosicoes.shape != (quantidadeDeMedicoes,)):
                        ProcessadorBase.__registrarErro(CODIGO_ERRO_FORMATO_BASE_INVALIDO,"O formato do vetor de posições é inválido","ProcessadorBase.validarBase",str(vetorPosicoes.shape))
                        return False

                    elif(vetorRepeticoes.shape != (quantidadeDeMedicoes,)):
                        ProcessadorBase.__registrarErro(CODIGO_ERRO_FORMATO_BASE_INVALIDO,"O formato do vetor de repetições é inválido","ProcessadorBase.validarBase",str(vetorRepeticoes.shape))
                        return False

                    elif(vetorGrupos.shape != (quantidadeDeMedicoes,)):
                        ProcessadorBase.__registrarErro(CODIGO_ERRO_FORMATO_BASE_INVALIDO,"O formato do vetor de grupos é inválido","ProcessadorBase.validarBase",str(vetorGrupos.shape))
                        return False

                    elif(vetorArquivos.shape != (quantidadeDeMedicoes,)):
                        ProcessadorBase.__registrarErro(CODIGO_ERRO_FORMATO_BASE_INVALIDO,"O formato do vetor de arquivos é inválido","ProcessadorBase.validarBase",str(vetorArquivos.shape))
                        return False

                    elif(not np.all(np.isfinite(vetorFrequencias))):
                        ProcessadorBase.__registrarErro(CODIGO_ERRO_CONTEUDO_BASE_INVALIDO,"O vetor de frequências contém valores inválidos","ProcessadorBase.validarBase")
                        return False

                    elif(not np.all(np.isfinite(matrizS21))):
                        ProcessadorBase.__registrarErro(CODIGO_ERRO_CONTEUDO_BASE_INVALIDO,"A matriz S21 contém valores inválidos","ProcessadorBase.validarBase")
                        return False

                    elif(np.any(vetorClasses < 0) or np.any(vetorClasses >= QUANTIDADE_AMBIENTES)):
                        ProcessadorBase.__registrarErro(CODIGO_ERRO_CONTEUDO_BASE_INVALIDO,"O vetor de classes contém valores inválidos","ProcessadorBase.validarBase")
                        return False

                    elif(np.any(vetorRepeticoes < 1) or np.any(vetorRepeticoes > QUANTIDADE_REPETICOES_POR_POSICAO)):
                        ProcessadorBase.__registrarErro(CODIGO_ERRO_CONTEUDO_BASE_INVALIDO,"O vetor de repetições contém valores inválidos","ProcessadorBase.validarBase")
                        return False

                    elif(np.any(vetorGrupos < 0) or np.any(vetorGrupos >= QUANTIDADE_GRUPOS_TOTAL)):
                        ProcessadorBase.__registrarErro(CODIGO_ERRO_CONTEUDO_BASE_INVALIDO,"O vetor de grupos contém valores inválidos","ProcessadorBase.validarBase")
                        return False

                    else:
                        frequenciasEsperadas = np.linspace(FREQUENCIA_INICIAL_HZ,FREQUENCIA_FINAL_HZ,QUANTIDADE_PONTOS_FREQUENCIA)

                        if(not np.allclose(vetorFrequencias,frequenciasEsperadas,rtol=0.0,atol=TOLERANCIA_FREQUENCIA_HZ)):
                            ProcessadorBase.__registrarErro(CODIGO_ERRO_CONTEUDO_BASE_INVALIDO,"A grade de frequências da base é inconsistente","ProcessadorBase.validarBase")
                            return False

                        elif(validarBaseCompleta):
                            if(quantidadeDeMedicoes != QUANTIDADE_MEDICOES_TOTAL):
                                ProcessadorBase.__registrarErro(CODIGO_ERRO_BASE_COMPLETA_INVALIDA,"A quantidade total de medições é inválida","ProcessadorBase.validarBase",str(quantidadeDeMedicoes))
                                return False

                            else:
                                contadorDeClasses = np.bincount(vetorClasses.astype(np.int64),minlength=QUANTIDADE_AMBIENTES)
                                gruposUnicos,contadorDeGrupos = np.unique(vetorGrupos,return_counts=True)

                                if(not np.all(contadorDeClasses == QUANTIDADE_MEDICOES_POR_AMBIENTE)):
                                    ProcessadorBase.__registrarErro(CODIGO_ERRO_BASE_COMPLETA_INVALIDA,"As classes não possuem a quantidade esperada de medições","ProcessadorBase.validarBase",str(contadorDeClasses.tolist()))
                                    return False

                                elif(gruposUnicos.size != QUANTIDADE_GRUPOS_TOTAL):
                                    ProcessadorBase.__registrarErro(CODIGO_ERRO_BASE_COMPLETA_INVALIDA,"A quantidade de grupos físicos é inválida","ProcessadorBase.validarBase",str(gruposUnicos.size))
                                    return False

                                elif(not np.all(contadorDeGrupos == QUANTIDADE_REPETICOES_POR_POSICAO)):
                                    ProcessadorBase.__registrarErro(CODIGO_ERRO_BASE_COMPLETA_INVALIDA,"Os grupos físicos não possuem a quantidade esperada de repetições","ProcessadorBase.validarBase")
                                    return False

                                else:
                                    return True

                        else:
                            return True

        except Exception as excecao:
            ProcessadorBase.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado durante a validação da base","ProcessadorBase.validarBase",str(excecao))
            return False

    @staticmethod
    def salvarBase(dicionarioBase:dict[str,np.ndarray],caminhoBaseProcessada:Path | str=CAMINHO_BASE_PROCESSADA,validarBaseCompleta:bool=False) -> bool:
        ProcessadorBase.__limparUltimoErro()

        try:
            if(not ProcessadorBase.validarBase(dicionarioBase,validarBaseCompleta)):
                return False

            else:
                caminhoBaseProcessada = Path(caminhoBaseProcessada)
                caminhoBaseProcessada.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(caminhoBaseProcessada,**dicionarioBase)
                return True

        except Exception as excecao:
            ProcessadorBase.__registrarErro(CODIGO_ERRO_SALVAMENTO_BASE,"Não foi possível salvar a base processada","ProcessadorBase.salvarBase",str(excecao))
            return False

    @staticmethod
    def carregarBase(caminhoBaseProcessada:Path | str=CAMINHO_BASE_PROCESSADA,validarBaseCompleta:bool=False) -> dict[str,np.ndarray] | None:
        ProcessadorBase.__limparUltimoErro()

        try:
            caminhoBaseProcessada = Path(caminhoBaseProcessada)

            if(not caminhoBaseProcessada.exists()):
                ProcessadorBase.__registrarErro(CODIGO_ERRO_CARREGAMENTO_BASE,"A base processada não foi encontrada","ProcessadorBase.carregarBase",str(caminhoBaseProcessada))
                return None

            elif(not caminhoBaseProcessada.is_file()):
                ProcessadorBase.__registrarErro(CODIGO_ERRO_CARREGAMENTO_BASE,"O caminho da base processada não representa um arquivo","ProcessadorBase.carregarBase",str(caminhoBaseProcessada))
                return None

            else:
                dicionarioBase = {}

                with np.load(caminhoBaseProcessada,allow_pickle=False) as arquivoBase:
                    for chave in arquivoBase.files:
                        dicionarioBase[chave] = arquivoBase[chave]

                if(not ProcessadorBase.validarBase(dicionarioBase,validarBaseCompleta)):
                    return None

                else:
                    return dicionarioBase

        except Exception as excecao:
            ProcessadorBase.__registrarErro(CODIGO_ERRO_CARREGAMENTO_BASE,"Não foi possível carregar a base processada","ProcessadorBase.carregarBase",str(excecao))
            return None

    @staticmethod
    def processarESalvar(caminhoArquivoZip:Path | str=CAMINHO_ARQUIVO_ZIP,caminhoBaseProcessada:Path | str=CAMINHO_BASE_PROCESSADA) -> bool:
        ProcessadorBase.__limparUltimoErro()

        try:
            dicionarioBase = ProcessadorBase.construirBase(caminhoArquivoZip)

            if(dicionarioBase is None):
                return False

            elif(not ProcessadorBase.salvarBase(dicionarioBase,caminhoBaseProcessada,True)):
                return False

            else:
                return True

        except Exception as excecao:
            ProcessadorBase.__registrarErro(CODIGO_ERRO_PROCESSAMENTO_BASE,"Não foi possível processar e salvar a base","ProcessadorBase.processarESalvar",str(excecao))
            return False