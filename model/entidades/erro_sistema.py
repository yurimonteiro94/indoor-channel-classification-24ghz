from __future__ import annotations


class ErroSistema:
    def __init__(self,codigo:int,mensagem:str,origem:str,detalhe:str | None):
        self.__codigo = codigo
        self.__mensagem = mensagem
        self.__origem = origem
        self.__detalhe = detalhe

    @staticmethod
    def criar(codigo:int,mensagem:str,origem:str,detalhe:str | None=None) -> ErroSistema | None:
        try:
            erro = ErroSistema(codigo,mensagem,origem,detalhe)

            if(erro.validar()):
                return erro

            else:
                return None

        except Exception:
            return None

    def validar(self) -> bool:
        try:
            if(not isinstance(self.__codigo,int) or isinstance(self.__codigo,bool)):
                return False

            elif(self.__codigo <= 0):
                return False

            elif(not isinstance(self.__mensagem,str)):
                return False

            elif(self.__mensagem.strip() == ""):
                return False

            elif(not isinstance(self.__origem,str)):
                return False

            elif(self.__origem.strip() == ""):
                return False

            elif(self.__detalhe is not None and not isinstance(self.__detalhe,str)):
                return False

            else:
                return True

        except Exception:
            return False

    def getCodigo(self) -> int:
        return self.__codigo

    def getMensagem(self) -> str:
        return self.__mensagem

    def getOrigem(self) -> str:
        return self.__origem

    def getDetalhe(self) -> str | None:
        return self.__detalhe

    def setCodigo(self,codigo:int) -> bool:
        try:
            if(not isinstance(codigo,int) or isinstance(codigo,bool)):
                return False

            elif(codigo <= 0):
                return False

            else:
                self.__codigo = codigo
                return True

        except Exception:
            return False

    def setMensagem(self,mensagem:str) -> bool:
        try:
            if(not isinstance(mensagem,str)):
                return False

            elif(mensagem.strip() == ""):
                return False

            else:
                self.__mensagem = mensagem
                return True

        except Exception:
            return False

    def setOrigem(self,origem:str) -> bool:
        try:
            if(not isinstance(origem,str)):
                return False

            elif(origem.strip() == ""):
                return False

            else:
                self.__origem = origem
                return True

        except Exception:
            return False

    def setDetalhe(self,detalhe:str | None) -> bool:
        try:
            if(detalhe is not None and not isinstance(detalhe,str)):
                return False

            else:
                self.__detalhe = detalhe
                return True

        except Exception:
            return False

    def obterDescricaoCompleta(self) -> str | None:
        try:
            descricao = "[" + str(self.__codigo) + "] " + self.__mensagem + " | origem: " + self.__origem

            if(self.__detalhe is not None and self.__detalhe.strip() != ""):
                descricao = descricao + " | detalhe: " + self.__detalhe

            return descricao

        except Exception:
            return None