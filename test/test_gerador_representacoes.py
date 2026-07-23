import unittest

import numpy as np

from services.constantes import CHAVE_S21
from services.constantes import QUANTIDADE_CANAIS_S21
from services.constantes import QUANTIDADE_PONTOS_FREQUENCIA
from services.gerador_representacoes import GeradorRepresentacoes
from test.utilitarios_teste import criarBaseSinteticaExperimentos


class TestGeradorRepresentacoes(unittest.TestCase):
    def setUp(self):
        self.dicionarioBase = criarBaseSinteticaExperimentos()
        self.matrizS21 = self.dicionarioBase[CHAVE_S21]

    def test_gerarRepresentacoes(self):
        representacaoCartesiana = GeradorRepresentacoes.gerarCartesiana(self.matrizS21)
        representacaoPolar = GeradorRepresentacoes.gerarPolar(self.matrizS21)
        representacaoTemporal = GeradorRepresentacoes.gerarTemporal(self.matrizS21)

        formatoEsperado = (self.matrizS21.shape[0],QUANTIDADE_PONTOS_FREQUENCIA,QUANTIDADE_CANAIS_S21)

        self.assertIsNotNone(representacaoCartesiana)
        self.assertIsNotNone(representacaoPolar)
        self.assertIsNotNone(representacaoTemporal)
        self.assertEqual(representacaoCartesiana.shape,formatoEsperado)
        self.assertEqual(representacaoPolar.shape,formatoEsperado)
        self.assertEqual(representacaoTemporal.shape,formatoEsperado)
        self.assertTrue(np.all(np.isfinite(representacaoCartesiana)))
        self.assertTrue(np.all(np.isfinite(representacaoPolar)))
        self.assertTrue(np.all(np.isfinite(representacaoTemporal)))
        self.assertTrue(np.allclose(representacaoCartesiana,self.matrizS21))

    def test_gerarVetorAtrasos(self):
        vetorAtrasos = GeradorRepresentacoes.obterVetorAtrasos()

        self.assertIsNotNone(vetorAtrasos)
        self.assertEqual(vetorAtrasos.shape,(QUANTIDADE_PONTOS_FREQUENCIA,))
        self.assertEqual(vetorAtrasos[0],0.0)
        self.assertTrue(np.all(np.diff(vetorAtrasos) > 0.0))

    def test_rejeitarMatrizInvalida(self):
        matrizInvalida = np.zeros((10,20),dtype=np.float32)
        resultado = GeradorRepresentacoes.gerarPolar(matrizInvalida)

        self.assertIsNone(resultado)
        self.assertTrue(GeradorRepresentacoes.ultimaExecucaoDeuErro())
        self.assertIsNotNone(GeradorRepresentacoes.getUltimoErro())


if(__name__ == "__main__"):
    unittest.main()