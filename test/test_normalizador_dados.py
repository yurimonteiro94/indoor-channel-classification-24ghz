import unittest

import numpy as np

from services.constantes import CHAVE_CLASSES
from services.constantes import CHAVE_DADOS_NORMALIZADOS
from services.constantes import CHAVE_DESVIOS
from services.constantes import CHAVE_GRUPOS
from services.constantes import CHAVE_INDICES_TREINO
from services.constantes import CHAVE_MEDIAS
from services.constantes import CHAVE_S21
from services.gerador_representacoes import GeradorRepresentacoes
from services.normalizador_dados import NormalizadorDados
from services.separador_base import SeparadorBase
from test.utilitarios_teste import criarBaseSinteticaExperimentos


class TestNormalizadorDados(unittest.TestCase):
    def setUp(self):
        self.dicionarioBase = criarBaseSinteticaExperimentos()
        self.divisao = SeparadorBase.criarDivisao(self.dicionarioBase[CHAVE_CLASSES],self.dicionarioBase[CHAVE_GRUPOS])
        self.representacao = GeradorRepresentacoes.gerarCartesiana(self.dicionarioBase[CHAVE_S21])

    def test_normalizarUsandoSomenteTreino(self):
        indicesTreino = self.divisao[CHAVE_INDICES_TREINO]
        resultado = NormalizadorDados.normalizarPorCanais(self.representacao,indicesTreino)

        self.assertIsNotNone(resultado)

        matrizNormalizada = resultado[CHAVE_DADOS_NORMALIZADOS]
        vetorMedias = resultado[CHAVE_MEDIAS]
        vetorDesvios = resultado[CHAVE_DESVIOS]

        mediasDoTreino = np.mean(matrizNormalizada[indicesTreino],axis=(0,1))
        desviosDoTreino = np.std(matrizNormalizada[indicesTreino],axis=(0,1))

        self.assertTrue(np.allclose(mediasDoTreino,np.zeros(2),atol=1.0E-5))
        self.assertTrue(np.allclose(desviosDoTreino,np.ones(2),atol=1.0E-5))
        self.assertTrue(np.all(vetorDesvios > 0.0))

        matrizAplicada = NormalizadorDados.aplicarNormalizacao(self.representacao,vetorMedias,vetorDesvios)

        self.assertIsNotNone(matrizAplicada)
        self.assertTrue(np.allclose(matrizAplicada,matrizNormalizada))

    def test_rejeitarIndicesInvalidos(self):
        indicesInvalidos = np.asarray([0,0,1],dtype=np.int64)
        resultado = NormalizadorDados.normalizarPorCanais(self.representacao,indicesInvalidos)

        self.assertIsNone(resultado)
        self.assertTrue(NormalizadorDados.ultimaExecucaoDeuErro())
        self.assertIsNotNone(NormalizadorDados.getUltimoErro())


if(__name__ == "__main__"):
    unittest.main()