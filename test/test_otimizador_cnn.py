import unittest

import numpy as np

from services.otimizador_cnn import OtimizadorCNN


class TestOtimizadorCNN(unittest.TestCase):
    def test_criarConfiguracaoPadraoAPartirDoVetor(self):
        vetorVariaveis = np.asarray([-3.0,-4.0,0.20,1.0,1.0,2.0],dtype=np.float64)

        configuracao = OtimizadorCNN.criarConfiguracaoAPartirDoVetor(vetorVariaveis)

        self.assertIsNotNone(configuracao)
        self.assertEqual(configuracao.getCanaisPrimeiraCamada(),16)
        self.assertEqual(configuracao.getCanaisSegundaCamada(),32)
        self.assertEqual(configuracao.getCanaisTerceiraCamada(),64)
        self.assertEqual(configuracao.getNeuroniosCamadaDensa(),64)
        self.assertEqual(configuracao.getTamanhoLote(),128)
        self.assertAlmostEqual(configuracao.getDropout(),0.20)
        self.assertAlmostEqual(configuracao.getTaxaAprendizado(),0.001)
        self.assertAlmostEqual(configuracao.getDecaimentoPesos(),0.0001)

    def test_criarConfiguracaoNoLimiteSuperior(self):
        vetorVariaveis = np.asarray([-2.3,-2.5,0.50,3.0,3.0,3.0],dtype=np.float64)

        configuracao = OtimizadorCNN.criarConfiguracaoAPartirDoVetor(vetorVariaveis)

        self.assertIsNotNone(configuracao)
        self.assertEqual(configuracao.getCanaisPrimeiraCamada(),32)
        self.assertEqual(configuracao.getCanaisSegundaCamada(),64)
        self.assertEqual(configuracao.getCanaisTerceiraCamada(),128)
        self.assertEqual(configuracao.getNeuroniosCamadaDensa(),128)
        self.assertEqual(configuracao.getTamanhoLote(),256)
        self.assertAlmostEqual(configuracao.getDropout(),0.50)

    def test_rejeitarVetorInvalido(self):
        vetorVariaveis = np.asarray([-3.0,-4.0,0.20],dtype=np.float64)

        configuracao = OtimizadorCNN.criarConfiguracaoAPartirDoVetor(vetorVariaveis)

        self.assertIsNone(configuracao)
        self.assertTrue(OtimizadorCNN.ultimaExecucaoDeuErro())
        self.assertIsNotNone(OtimizadorCNN.getUltimoErro())


if(__name__ == "__main__"):
    unittest.main()