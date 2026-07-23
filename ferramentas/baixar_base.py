from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Callable
from urllib.request import Request
from urllib.request import urlopen

from model.entidades.erro_sistema import ErroSistema
from services.constantes import CAMINHO_ARQUIVO_ZIP
from services.constantes import CODIGO_ERRO_ARGUMENTO_INVALIDO
from services.constantes import CODIGO_ERRO_ARQUIVO_BAIXADO_INVALIDO
from services.constantes import CODIGO_ERRO_CRIACAO_DIRETORIO
from services.constantes import CODIGO_ERRO_DESCONHECIDO
from services.constantes import CODIGO_ERRO_DOWNLOAD_BASE
from services.constantes import CODIGO_ERRO_REMOCAO_ARQUIVO
from services.constantes import CODIGO_ERRO_URL_INVALIDA
from services.constantes import TAMANHO_BLOCO_DOWNLOAD_BYTES
from services.constantes import URL_BASE_UCI


class BaixadorBase:
    __ultimoErro = None

    @staticmethod
    def __registrarErro(codigo:int,mensagem:str,origem:str,detalhe:str | None=None) -> None:
        BaixadorBase.__ultimoErro = ErroSistema.criar(codigo,mensagem,origem,detalhe)

    @staticmethod
    def __limparUltimoErro() -> None:
        BaixadorBase.__ultimoErro = None

    @staticmethod
    def getUltimoErro() -> ErroSistema | None:
        return BaixadorBase.__ultimoErro

    @staticmethod
    def ultimaExecucaoDeuErro() -> bool:
        if(BaixadorBase.__ultimoErro is None):
            return False

        else:
            return True

    @staticmethod
    def __removerArquivo(caminhoArquivo:Path) -> bool:
        try:
            if(caminhoArquivo.exists()):
                caminhoArquivo.unlink()
                return True

            else:
                return True

        except Exception as excecao:
            BaixadorBase.__registrarErro(CODIGO_ERRO_REMOCAO_ARQUIVO,"Não foi possível remover um arquivo temporário","BaixadorBase.__removerArquivo",str(excecao))
            return False

    @staticmethod
    def __validarArquivoZip(caminhoArquivo:Path) -> bool:
        try:
            if(not caminhoArquivo.exists()):
                return False

            elif(not caminhoArquivo.is_file()):
                return False

            elif(not zipfile.is_zipfile(caminhoArquivo)):
                return False

            else:
                return True

        except Exception:
            return False

    @staticmethod
    def baixar(caminhoDestino:Path | str=CAMINHO_ARQUIVO_ZIP,urlBase:str=URL_BASE_UCI,observadorProgresso:Callable[[int,int],None] | None=None) -> Path | None:
        BaixadorBase.__limparUltimoErro()

        try:
            if(not isinstance(urlBase,str)):
                BaixadorBase.__registrarErro(CODIGO_ERRO_URL_INVALIDA,"A URL da base é inválida","BaixadorBase.baixar")
                return None

            elif(urlBase.strip() == ""):
                BaixadorBase.__registrarErro(CODIGO_ERRO_URL_INVALIDA,"A URL da base não pode ser vazia","BaixadorBase.baixar")
                return None

            elif(not isinstance(caminhoDestino,(Path,str))):
                BaixadorBase.__registrarErro(CODIGO_ERRO_ARGUMENTO_INVALIDO,"O caminho de destino é inválido","BaixadorBase.baixar")
                return None

            else:
                caminhoDestino = Path(caminhoDestino)
                caminhoTemporario = Path(str(caminhoDestino) + ".part")

                if(BaixadorBase.__validarArquivoZip(caminhoDestino)):
                    return caminhoDestino

                elif(caminhoDestino.exists()):
                    if(not BaixadorBase.__removerArquivo(caminhoDestino)):
                        return None

                    else:
                        BaixadorBase.__limparUltimoErro()

                if(caminhoTemporario.exists()):
                    if(not BaixadorBase.__removerArquivo(caminhoTemporario)):
                        return None

                    else:
                        BaixadorBase.__limparUltimoErro()

                try:
                    caminhoDestino.parent.mkdir(parents=True,exist_ok=True)

                except Exception as excecao:
                    BaixadorBase.__registrarErro(CODIGO_ERRO_CRIACAO_DIRETORIO,"Não foi possível criar o diretório de destino","BaixadorBase.baixar",str(excecao))
                    return None

                requisicao = Request(urlBase,headers={"User-Agent":"Mozilla/5.0"})

                with urlopen(requisicao) as resposta:
                    valorTamanhoTotal = resposta.headers.get("Content-Length")

                    if(valorTamanhoTotal is None):
                        tamanhoTotal = 0

                    else:
                        tamanhoTotal = int(valorTamanhoTotal)

                    quantidadeBaixada = 0

                    with open(caminhoTemporario,"wb") as arquivoSaida:
                        while(True):
                            bloco = resposta.read(TAMANHO_BLOCO_DOWNLOAD_BYTES)

                            if(not bloco):
                                break

                            else:
                                arquivoSaida.write(bloco)
                                quantidadeBaixada = quantidadeBaixada + len(bloco)

                                if(observadorProgresso is not None):
                                    observadorProgresso(quantidadeBaixada,tamanhoTotal)

                if(not BaixadorBase.__validarArquivoZip(caminhoTemporario)):
                    BaixadorBase.__removerArquivo(caminhoTemporario)
                    BaixadorBase.__registrarErro(CODIGO_ERRO_ARQUIVO_BAIXADO_INVALIDO,"O arquivo baixado não é um ZIP válido","BaixadorBase.baixar",str(caminhoTemporario))
                    return None

                else:
                    os.replace(caminhoTemporario,caminhoDestino)

                    if(not BaixadorBase.__validarArquivoZip(caminhoDestino)):
                        BaixadorBase.__registrarErro(CODIGO_ERRO_ARQUIVO_BAIXADO_INVALIDO,"O arquivo final não é um ZIP válido","BaixadorBase.baixar",str(caminhoDestino))
                        return None

                    else:
                        BaixadorBase.__limparUltimoErro()
                        return caminhoDestino

        except Exception as excecao:
            try:
                caminhoTemporario = Path(str(caminhoDestino) + ".part")
                BaixadorBase.__removerArquivo(caminhoTemporario)

            except Exception:
                pass

            BaixadorBase.__registrarErro(CODIGO_ERRO_DOWNLOAD_BASE,"Não foi possível baixar a base","BaixadorBase.baixar",str(excecao))
            return None


def exibirProgresso(quantidadeBaixada:int,tamanhoTotal:int) -> None:
    if(tamanhoTotal > 0):
        percentual = 100.0 * quantidadeBaixada / tamanhoTotal
        print("\rDownload: %.1f%%" % percentual,end="",flush=True)

    else:
        quantidadeMegabytes = quantidadeBaixada / 1048576.0
        print("\rDownload: %.2f MB" % quantidadeMegabytes,end="",flush=True)


def executar() -> bool:
    caminhoArquivo = BaixadorBase.baixar(CAMINHO_ARQUIVO_ZIP,URL_BASE_UCI,exibirProgresso)

    if(caminhoArquivo is None):
        erro = BaixadorBase.getUltimoErro()

        if(erro is None):
            print("Não foi possível baixar a base.")

        else:
            descricao = erro.obterDescricaoCompleta()

            if(descricao is None):
                print("Não foi possível obter a descrição do erro.")

            else:
                print()
                print(descricao)

        return False

    else:
        print()
        print("Base disponível em: " + str(caminhoArquivo))
        return True


if(__name__ == "__main__"):
    executar()