import tempfile
import unittest
from pathlib import Path

import numpy as np

from services.constantes import CHAVE_CLASSES
from services.constantes import CHAVE_INDICES_TESTE
from services.constantes import CHAVE_INDICES_TREINO
from services.constantes import CHAVE_INDICES_VALIDACAO
from services.constantes import CHAVE_REPRESENTACAO_CARTESIANA
from services.constantes import CHAVE_REPRESENTACAO_POLAR
from services.constantes import CHAVE_REPRESENTACAO_TEMPORAL
from services.preparador_experimentos import PreparadorExperimentos
from test.utilitarios_teste import criarBaseSinteticaExperimentos


class TestPreparadorExperimentos(unittest.TestCase):
    def setUp(self):
        self.dicionarioBase = criarBaseSinteticaExperimentos()

    def test_prepararBase(self):
        basePreparada = PreparadorExperimentos.prepararBase(self.dicionarioBase)

        self.assertIsNotNone(basePreparada)
        self.assertTrue(PreparadorExperimentos.validarBasePreparada(basePreparada))

        quantidadeDeMedicoes = basePreparada[CHAVE_CLASSES].size

        self.assertEqual(basePreparada[CHAVE_REPRESENTACAO_CARTESIANA].shape[0],quantidadeDeMedicoes)
        self.assertEqual(basePreparada[CHAVE_REPRESENTACAO_POLAR].shape[0],quantidadeDeMedicoes)
        self.assertEqual(basePreparada[CHAVE_REPRESENTACAO_TEMPORAL].shape[0],quantidadeDeMedicoes)
        self.assertEqual(basePreparada[CHAVE_INDICES_TREINO].size,56)
        self.assertEqual(basePreparada[CHAVE_INDICES_VALIDACAO].size,8)
        self.assertEqual(basePreparada[CHAVE_INDICES_TESTE].size,16)

    def test_salvarECarregarBasePreparada(self):
        with tempfile.TemporaryDirectory() as diretorioTemporario:
            caminhoArquivo = Path(diretorioTemporario) / "base_preparada.npz"
            basePreparada = PreparadorExperimentos.prepararBase(self.dicionarioBase)

            self.assertIsNotNone(basePreparada)
            self.assertTrue(PreparadorExperimentos.salvarBasePreparada(basePreparada,caminhoArquivo))

            baseCarregada = PreparadorExperimentos.carregarBasePreparada(caminhoArquivo)

            self.assertIsNotNone(baseCarregada)
            self.assertTrue(PreparadorExperimentos.validarBasePreparada(baseCarregada))
            self.assertTrue(np.array_equal(basePreparada[CHAVE_INDICES_TREINO],baseCarregada[CHAVE_INDICES_TREINO]))
            self.assertTrue(np.allclose(basePreparada[CHAVE_REPRESENTACAO_POLAR],baseCarregada[CHAVE_REPRESENTACAO_POLAR]))


if(__name__ == "__main__"):
    unittest.main()