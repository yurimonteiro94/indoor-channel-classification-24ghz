from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path

import numpy as np

from model.entidades.erro_sistema import ErroSistema
from model.entidades.medicao_canal import MedicaoCanal
from services.constantes import CAMINHO_ARQUIVO_ZIP
from services.constantes import CODIFICACAO_ARQUIVOS
from services.constantes import CODIGO_ERRO_AMBIENTE_DESCONHECIDO
from services.constantes import CODIGO_ERRO_ARGUMENTO_INVALIDO
from services.constantes import CODIGO_ERRO_ARQUIVO_NAO_ENCONTRADO
from services.constantes import CODIGO_ERRO_ARQUIVO_ZIP_INVALIDO
from services.constantes import CODIGO_ERRO_CAMINHO_INTERNO_INVALIDO
from services.constantes import CODIGO_ERRO_CAMINHO_NAO_E_ARQUIVO
from services.constantes import CODIGO_ERRO_DESCONHECIDO
from services.constantes import CODIGO_ERRO_LEITURA_MEDICAO
from services.constantes import CODIGO_ERRO_LINHA_CSV_INVALIDA
from services.constantes import CODIGO_ERRO_NENHUMA_MEDICAO_ENCONTRADA
from services.constantes import CODIGO_ERRO_POSICAO_INVALIDA
from services.constantes import CODIGO_ERRO_REPETICAO_INVALIDA
from services.constantes import DICIONARIO_CLASSES_POR_DIRETORIO
from services.constantes import EXTENSAO_ARQUIVO_MEDICAO
from services.constantes import INDICE_COLUNA_FREQUENCIA
from services.constantes import INDICE_COLUNA_IMAGINARIA_S21
from services.constantes import INDICE_COLUNA_REAL_S21
from services.constantes import PREFIXO_DIRETORIO_MEDICOES
from services.constantes import PREFIXO_LINHA_COMENTARIO
from services.constantes import QUANTIDADE_POSICOES_POR_AMBIENTE
from services.constantes import QUANTIDADE_LINHAS_POSICOES
from services.constantes import QUANTIDADE_COLUNAS_POSICOES
from services.constantes import QUANTIDADE_REPETICOES_POR_POSICAO
from services.constantes import SEPARADOR_CSV


class MedicaoCanalDAO:
    __ultimoErro = None

    @staticmethod
    def __registrarErro(codigo:int,mensagem:str,origem:str,detalhe:str | None=None) -> None:
        MedicaoCanalDAO.__ultimoErro = ErroSistema.criar(codigo,mensagem,origem,detalhe)

    @staticmethod
    def __limparUltimoErro() -> None:
        MedicaoCanalDAO.__ultimoErro = None

    @staticmethod
    def getUltimoErro() -> ErroSistema | None:
        return MedicaoCanalDAO.__ultimoErro

    @staticmethod
    def ultimaExecucaoDeuErro() -> bool:
        if(MedicaoCanalDAO.__ultimoErro is None):
            return False

        else:
            return True

    @staticmethod
    def __validarArquivoZip(caminhoArquivoZip:Path) -> bool:
        try:
            if(not caminhoArquivoZip.exists()):
                MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_ARQUIVO_NAO_ENCONTRADO,"O arquivo ZIP não foi encontrado","MedicaoCanalDAO.__validarArquivoZip",str(caminhoArquivoZip))
                return False

            elif(not caminhoArquivoZip.is_file()):
                MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_CAMINHO_NAO_E_ARQUIVO,"O caminho informado não representa um arquivo","MedicaoCanalDAO.__validarArquivoZip",str(caminhoArquivoZip))
                return False

            elif(not zipfile.is_zipfile(caminhoArquivoZip)):
                MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_ARQUIVO_ZIP_INVALIDO,"O arquivo informado não é um ZIP válido","MedicaoCanalDAO.__validarArquivoZip",str(caminhoArquivoZip))
                return False

            else:
                return True

        except Exception as excecao:
            MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado ao validar o arquivo ZIP","MedicaoCanalDAO.__validarArquivoZip",str(excecao))
            return False

    @staticmethod
    def listarCaminhosMedicoes(caminhoArquivoZip:Path | str=CAMINHO_ARQUIVO_ZIP) -> list[str] | None:
        MedicaoCanalDAO.__limparUltimoErro()

        try:
            caminhoArquivoZip = Path(caminhoArquivoZip)

            if(not MedicaoCanalDAO.__validarArquivoZip(caminhoArquivoZip)):
                return None

            listaDeCaminhos = []

            with zipfile.ZipFile(caminhoArquivoZip,"r") as arquivoZip:
                for caminho in arquivoZip.namelist():
                    caminhoNormalizado = caminho.replace("\\","/")

                    if(caminhoNormalizado.lower().endswith(EXTENSAO_ARQUIVO_MEDICAO)):
                        if(caminhoNormalizado.startswith(PREFIXO_DIRETORIO_MEDICOES)):
                            listaDeCaminhos.append(caminhoNormalizado)

            if(len(listaDeCaminhos) == 0):
                MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_NENHUMA_MEDICAO_ENCONTRADA,"Nenhum arquivo de medição foi encontrado no ZIP","MedicaoCanalDAO.listarCaminhosMedicoes",str(caminhoArquivoZip))
                return None

            else:
                listaDeCaminhos.sort(key=MedicaoCanalDAO.__obterChaveOrdenacao)

                if(MedicaoCanalDAO.ultimaExecucaoDeuErro()):
                    return None

                else:
                    return listaDeCaminhos

        except Exception as excecao:
            MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado ao listar os arquivos de medição","MedicaoCanalDAO.listarCaminhosMedicoes",str(excecao))
            return None

    @staticmethod
    def contarMedicoes(caminhoArquivoZip:Path | str=CAMINHO_ARQUIVO_ZIP) -> int | None:
        MedicaoCanalDAO.__limparUltimoErro()

        try:
            listaDeCaminhos = MedicaoCanalDAO.listarCaminhosMedicoes(caminhoArquivoZip)

            if(listaDeCaminhos is None):
                return None

            else:
                return len(listaDeCaminhos)

        except Exception as excecao:
            MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado ao contar as medições","MedicaoCanalDAO.contarMedicoes",str(excecao))
            return None

    @staticmethod
    def carregarMedicaoPorCaminho(caminhoInterno:str,caminhoArquivoZip:Path | str=CAMINHO_ARQUIVO_ZIP) -> MedicaoCanal | None:
        MedicaoCanalDAO.__limparUltimoErro()

        try:
            caminhoArquivoZip = Path(caminhoArquivoZip)

            if(not isinstance(caminhoInterno,str) or caminhoInterno.strip() == ""):
                MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_ARGUMENTO_INVALIDO,"O caminho interno da medição é inválido","MedicaoCanalDAO.carregarMedicaoPorCaminho")
                return None

            elif(not MedicaoCanalDAO.__validarArquivoZip(caminhoArquivoZip)):
                return None

            else:
                with zipfile.ZipFile(caminhoArquivoZip,"r") as arquivoZip:
                    if(caminhoInterno not in arquivoZip.namelist()):
                        MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_CAMINHO_INTERNO_INVALIDO,"O arquivo de medição não existe dentro do ZIP","MedicaoCanalDAO.carregarMedicaoPorCaminho",caminhoInterno)
                        return None

                    else:
                        return MedicaoCanalDAO.__lerMedicaoDoZip(arquivoZip,caminhoInterno)

        except Exception as excecao:
            MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_LEITURA_MEDICAO,"Não foi possível carregar a medição","MedicaoCanalDAO.carregarMedicaoPorCaminho",str(excecao))
            return None

    @staticmethod
    def carregarMedicoes(caminhoArquivoZip:Path | str=CAMINHO_ARQUIVO_ZIP,quantidadeMaxima:int | None=None) -> list[MedicaoCanal] | None:
        MedicaoCanalDAO.__limparUltimoErro()

        try:
            listaDeCaminhos = MedicaoCanalDAO.listarCaminhosMedicoes(caminhoArquivoZip)

            if(listaDeCaminhos is None):
                return None

            elif(quantidadeMaxima is not None and (not isinstance(quantidadeMaxima,int) or isinstance(quantidadeMaxima,bool))):
                MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_ARGUMENTO_INVALIDO,"A quantidade máxima de medições é inválida","MedicaoCanalDAO.carregarMedicoes")
                return None

            elif(quantidadeMaxima is not None and quantidadeMaxima <= 0):
                MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_ARGUMENTO_INVALIDO,"A quantidade máxima de medições deve ser positiva","MedicaoCanalDAO.carregarMedicoes")
                return None

            else:
                if(quantidadeMaxima is not None):
                    listaDeCaminhos = listaDeCaminhos[:quantidadeMaxima]

                listaDeMedicoes = []
                caminhoArquivoZip = Path(caminhoArquivoZip)

                with zipfile.ZipFile(caminhoArquivoZip,"r") as arquivoZip:
                    for caminhoInterno in listaDeCaminhos:
                        medicao = MedicaoCanalDAO.__lerMedicaoDoZip(arquivoZip,caminhoInterno)

                        if(medicao is None):
                            return None

                        else:
                            listaDeMedicoes.append(medicao)

                return listaDeMedicoes

        except Exception as excecao:
            MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_LEITURA_MEDICAO,"Não foi possível carregar as medições","MedicaoCanalDAO.carregarMedicoes",str(excecao))
            return None

    @staticmethod
    def __obterChaveOrdenacao(caminhoInterno:str) -> tuple[int,int,int]:
        metadados = MedicaoCanalDAO.__extrairMetadados(caminhoInterno)

        if(metadados is None):
            return 9999,9999,9999

        else:
            return metadados[1],metadados[3],metadados[4]

    @staticmethod
    def __extrairMetadados(caminhoInterno:str) -> tuple[str,int,str,int,int,int] | None:
        try:
            caminhoNormalizado = caminhoInterno.replace("\\","/")

            if(not caminhoNormalizado.startswith(PREFIXO_DIRETORIO_MEDICOES)):
                MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_CAMINHO_INTERNO_INVALIDO,"O arquivo está fora do diretório esperado","MedicaoCanalDAO.__extrairMetadados",caminhoInterno)
                return None

            else:
                caminhoRelativo = caminhoNormalizado[len(PREFIXO_DIRETORIO_MEDICOES):]
                partes = caminhoRelativo.split("/")

                if(len(partes) != 3):
                    MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_CAMINHO_INTERNO_INVALIDO,"A estrutura do caminho interno é inválida","MedicaoCanalDAO.__extrairMetadados",caminhoInterno)
                    return None

                else:
                    nomeAmbiente = partes[0]
                    posicao = partes[1]
                    nomeArquivo = partes[2]

                    if(nomeAmbiente not in DICIONARIO_CLASSES_POR_DIRETORIO):
                        MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_AMBIENTE_DESCONHECIDO,"O ambiente da medição é desconhecido","MedicaoCanalDAO.__extrairMetadados",nomeAmbiente)
                        return None

                    else:
                        correspondenciaPosicao = re.fullmatch(r"Loc_(\d{4})",posicao)

                        if(correspondenciaPosicao is None):
                            MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_POSICAO_INVALIDA,"O identificador da posição é inválido","MedicaoCanalDAO.__extrairMetadados",posicao)
                            return None

                        else:
                            codigoPosicao = correspondenciaPosicao.group(1)
                            linhaPosicao = int(codigoPosicao[:2])
                            colunaPosicao = int(codigoPosicao[2:])
                            indicePosicao = linhaPosicao * QUANTIDADE_COLUNAS_POSICOES + colunaPosicao

                            if(linhaPosicao < 0 or linhaPosicao >= QUANTIDADE_LINHAS_POSICOES or colunaPosicao < 0 or colunaPosicao >= QUANTIDADE_COLUNAS_POSICOES):
                                MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_POSICAO_INVALIDA,"A posição está fora da grade esperada de 14 por 14","MedicaoCanalDAO.__extrairMetadados",posicao)
                                return None

                            else:
                                correspondenciaRepeticao = re.search(r"_(\d+)Ch1\.csv$",nomeArquivo,re.IGNORECASE)

                                if(correspondenciaRepeticao is None):
                                    MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_REPETICAO_INVALIDA,"O número da repetição não foi encontrado no nome do arquivo","MedicaoCanalDAO.__extrairMetadados",nomeArquivo)
                                    return None

                                else:
                                    repeticao = int(correspondenciaRepeticao.group(1))

                                    if(repeticao < 1 or repeticao > QUANTIDADE_REPETICOES_POR_POSICAO):
                                        MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_REPETICAO_INVALIDA,"O número da repetição está fora do intervalo esperado","MedicaoCanalDAO.__extrairMetadados",nomeArquivo)
                                        return None

                                    else:
                                        classe = DICIONARIO_CLASSES_POR_DIRETORIO[nomeAmbiente]
                                        grupo = classe * QUANTIDADE_POSICOES_POR_AMBIENTE + indicePosicao
                                        return nomeAmbiente,classe,posicao,indicePosicao,repeticao,grupo

        except Exception as excecao:
            MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_DESCONHECIDO,"Erro inesperado ao extrair os metadados","MedicaoCanalDAO.__extrairMetadados",str(excecao))
            return None

    @staticmethod
    def __lerMedicaoDoZip(arquivoZip:zipfile.ZipFile,caminhoInterno:str) -> MedicaoCanal | None:
        try:
            metadados = MedicaoCanalDAO.__extrairMetadados(caminhoInterno)

            if(metadados is None):
                return None

            else:
                nomeAmbiente = metadados[0]
                classe = metadados[1]
                posicao = metadados[2]
                repeticao = metadados[4]
                grupo = metadados[5]

                listaDeFrequencias = []
                listaDeParteRealS21 = []
                listaDeParteImaginariaS21 = []

                with arquivoZip.open(caminhoInterno,"r") as arquivoBinario:
                    with io.TextIOWrapper(arquivoBinario,encoding=CODIFICACAO_ARQUIVOS,newline="") as arquivoTexto:
                        leitorCsv = csv.reader(arquivoTexto,delimiter=SEPARADOR_CSV)

                        for linha in leitorCsv:
                            if(len(linha) == 0):
                                continue

                            else:
                                primeiroCampo = linha[0].strip()

                                if(primeiroCampo == ""):
                                    continue

                                elif(primeiroCampo.startswith(PREFIXO_LINHA_COMENTARIO)):
                                    continue

                                elif(primeiroCampo.lower().startswith("freq")):
                                    continue

                                elif(len(linha) <= INDICE_COLUNA_IMAGINARIA_S21):
                                    MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_LINHA_CSV_INVALIDA,"Uma linha do arquivo de medição está incompleta","MedicaoCanalDAO.__lerMedicaoDoZip",caminhoInterno)
                                    return None

                                else:
                                    listaDeFrequencias.append(float(linha[INDICE_COLUNA_FREQUENCIA]))
                                    listaDeParteRealS21.append(float(linha[INDICE_COLUNA_REAL_S21]))
                                    listaDeParteImaginariaS21.append(float(linha[INDICE_COLUNA_IMAGINARIA_S21]))

                medicao = MedicaoCanal.criar(nomeAmbiente,classe,posicao,repeticao,grupo,caminhoInterno,np.asarray(listaDeFrequencias,dtype=np.float64),np.asarray(listaDeParteRealS21,dtype=np.float64),np.asarray(listaDeParteImaginariaS21,dtype=np.float64))

                if(medicao is None):
                    MedicaoCanalDAO.__ultimoErro = MedicaoCanal.getUltimoErro()
                    return None

                else:
                    return medicao

        except Exception as excecao:
            MedicaoCanalDAO.__registrarErro(CODIGO_ERRO_LEITURA_MEDICAO,"Não foi possível interpretar o arquivo de medição","MedicaoCanalDAO.__lerMedicaoDoZip",str(excecao))
            return None