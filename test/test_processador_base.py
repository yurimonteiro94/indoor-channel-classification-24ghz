import tempfile
import unittest
from pathlib import Path

import numpy as np

from services.constantes import CHAVE_CLASSES
from services.constantes import CHAVE_GRUPOS
from services.constantes import CHAVE_S21
from services.constantes import QUANTIDADE_CANAIS_S21
from services.constantes import QUANTIDADE_PONTOS_FREQUENCIA
from services.processador_base import ProcessadorBase
from test.utilitarios_teste import criarZipSintetico


class TestProcessadorBase(unittest.TestCase):
    def setUp(self):
        self.diretorioTemporario = tempfile.TemporaryDirectory()
        self.caminhoArquivoZip = criarZipSintetico(self.diretorioTemporario.name)
        self.caminhoBaseProcessada = Path(self.diretorioTemporario.name) / "base_processada.npz"

    def tearDown(self):
        self.diretorioTemporario.cleanup()

    def test_construirBase(self):
        dicionarioBase = ProcessadorBase.construirBase(self.caminhoArquivoZip)

        self.assertIsNotNone(dicionarioBase)
        self.assertEqual(dicionarioBase[CHAVE_S21].shape,(4,QUANTIDADE_PONTOS_FREQUENCIA,QUANTIDADE_CANAIS_S21))
        self.assertEqual(dicionarioBase[CHAVE_CLASSES].tolist(),[0,0,1,1])
        self.assertEqual(dicionarioBase[CHAVE_GRUPOS].tolist(),[0,0,197,197])
        self.assertTrue(ProcessadorBase.validarBase(dicionarioBase))
        self.assertFalse(ProcessadorBase.validarBase(dicionarioBase,True))

    def test_salvarECarregarBase(self):
        dicionarioBaseOriginal = ProcessadorBase.construirBase(self.caminhoArquivoZip)

        self.assertIsNotNone(dicionarioBaseOriginal)
        self.assertTrue(ProcessadorBase.salvarBase(dicionarioBaseOriginal,self.caminhoBaseProcessada))
        self.assertTrue(self.caminhoBaseProcessada.exists())

        dicionarioBaseCarregada = ProcessadorBase.carregarBase(self.caminhoBaseProcessada)

        self.assertIsNotNone(dicionarioBaseCarregada)
        self.assertTrue(np.array_equal(dicionarioBaseOriginal[CHAVE_CLASSES],dicionarioBaseCarregada[CHAVE_CLASSES]))
        self.assertTrue(np.allclose(dicionarioBaseOriginal[CHAVE_S21],dicionarioBaseCarregada[CHAVE_S21]))

    def test_rejeitarFormatoInvalido(self):
        dicionarioBase = ProcessadorBase.construirBase(self.caminhoArquivoZip)

        self.assertIsNotNone(dicionarioBase)

        dicionarioBase[CHAVE_S21] = np.zeros((4,10,2),dtype=np.float32)

        self.assertFalse(ProcessadorBase.validarBase(dicionarioBase))
        self.assertTrue(ProcessadorBase.ultimaExecucaoDeuErro())


if(__name__ == "__main__"):
    unittest.main()