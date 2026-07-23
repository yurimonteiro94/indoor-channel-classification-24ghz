import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from ferramentas.baixar_base import BaixadorBase
from services.constantes import CODIGO_ERRO_URL_INVALIDA


class RespostaFalsa:
    def __init__(self,conteudo:bytes):
        self.__fluxo = io.BytesIO(conteudo)
        self.headers = {"Content-Length":str(len(conteudo))}

    def __enter__(self):
        return self

    def __exit__(self,tipoExcecao,valorExcecao,rastreamento):
        self.__fluxo.close()
        return False

    def read(self,tamanho:int) -> bytes:
        return self.__fluxo.read(tamanho)


class TestBaixadorBase(unittest.TestCase):
    def criarConteudoZip(self) -> bytes:
        memoria = io.BytesIO()

        with zipfile.ZipFile(memoria,"w",compression=zipfile.ZIP_DEFLATED) as arquivoZip:
            arquivoZip.writestr("arquivo_teste.txt","conteudo de teste")

        return memoria.getvalue()

    def test_baixarArquivoValido(self):
        with tempfile.TemporaryDirectory() as diretorioTemporario:
            caminhoDestino = Path(diretorioTemporario) / "base.zip"
            resposta = RespostaFalsa(self.criarConteudoZip())

            with patch("ferramentas.baixar_base.urlopen",return_value=resposta):
                resultado = BaixadorBase.baixar(caminhoDestino,"https://exemplo.com/base.zip")

            self.assertIsNotNone(resultado)
            self.assertTrue(caminhoDestino.exists())
            self.assertTrue(zipfile.is_zipfile(caminhoDestino))
            self.assertFalse(BaixadorBase.ultimaExecucaoDeuErro())
            self.assertIsNone(BaixadorBase.getUltimoErro())

    def test_reutilizarArquivoExistente(self):
        with tempfile.TemporaryDirectory() as diretorioTemporario:
            caminhoDestino = Path(diretorioTemporario) / "base.zip"
            caminhoDestino.write_bytes(self.criarConteudoZip())

            with patch("ferramentas.baixar_base.urlopen") as urlopenSimulado:
                resultado = BaixadorBase.baixar(caminhoDestino,"https://exemplo.com/base.zip")

            self.assertIsNotNone(resultado)
            self.assertEqual(resultado,caminhoDestino)
            urlopenSimulado.assert_not_called()

    def test_rejeitarUrlInvalida(self):
        with tempfile.TemporaryDirectory() as diretorioTemporario:
            caminhoDestino = Path(diretorioTemporario) / "base.zip"
            resultado = BaixadorBase.baixar(caminhoDestino,"")
            erro = BaixadorBase.getUltimoErro()

            self.assertIsNone(resultado)
            self.assertTrue(BaixadorBase.ultimaExecucaoDeuErro())
            self.assertIsNotNone(erro)
            self.assertEqual(erro.getCodigo(),CODIGO_ERRO_URL_INVALIDA)


if(__name__ == "__main__"):
    unittest.main()