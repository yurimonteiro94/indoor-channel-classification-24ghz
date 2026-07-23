import unittest

import numpy as np

from services.constantes import CHAVE_CLASSES
from services.constantes import CHAVE_GRUPOS
from services.constantes import CHAVE_GRUPOS_TESTE
from services.constantes import CHAVE_GRUPOS_TREINO
from services.constantes import CHAVE_GRUPOS_VALIDACAO
from services.constantes import CHAVE_INDICES_TESTE
from services.constantes import CHAVE_INDICES_TREINO
from services.constantes import CHAVE_INDICES_VALIDACAO
from services.constantes import QUANTIDADE_AMBIENTES
from services.separador_base import SeparadorBase
from test.utilitarios_teste import criarBaseSinteticaExperimentos


class TestSeparadorBase(unittest.TestCase):
    def setUp(self):
        self.dicionarioBase = criarBaseSinteticaExperimentos()

    def test_criarDivisaoSemVazamento(self):
        vetorClasses = self.dicionarioBase[CHAVE_CLASSES]
        vetorGrupos = self.dicionarioBase[CHAVE_GRUPOS]
        divisao = SeparadorBase.criarDivisao(vetorClasses,vetorGrupos)

        self.assertIsNotNone(divisao)
        self.assertTrue(SeparadorBase.validarDivisao(divisao,vetorClasses,vetorGrupos))
        self.assertEqual(divisao[CHAVE_INDICES_TREINO].size,56)
        self.assertEqual(divisao[CHAVE_INDICES_VALIDACAO].size,8)
        self.assertEqual(divisao[CHAVE_INDICES_TESTE].size,16)

        gruposTreino = set(divisao[CHAVE_GRUPOS_TREINO].tolist())
        gruposValidacao = set(divisao[CHAVE_GRUPOS_VALIDACAO].tolist())
        gruposTeste = set(divisao[CHAVE_GRUPOS_TESTE].tolist())

        self.assertTrue(gruposTreino.isdisjoint(gruposValidacao))
        self.assertTrue(gruposTreino.isdisjoint(gruposTeste))
        self.assertTrue(gruposValidacao.isdisjoint(gruposTeste))

        classesTreino = np.bincount(vetorClasses[divisao[CHAVE_INDICES_TREINO]].astype(np.int64),minlength=QUANTIDADE_AMBIENTES)
        classesValidacao = np.bincount(vetorClasses[divisao[CHAVE_INDICES_VALIDACAO]].astype(np.int64),minlength=QUANTIDADE_AMBIENTES)
        classesTeste = np.bincount(vetorClasses[divisao[CHAVE_INDICES_TESTE]].astype(np.int64),minlength=QUANTIDADE_AMBIENTES)

        self.assertEqual(classesTreino.tolist(),[14,14,14,14])
        self.assertEqual(classesValidacao.tolist(),[2,2,2,2])
        self.assertEqual(classesTeste.tolist(),[4,4,4,4])

    def test_divisaoReproduzivel(self):
        vetorClasses = self.dicionarioBase[CHAVE_CLASSES]
        vetorGrupos = self.dicionarioBase[CHAVE_GRUPOS]

        primeiraDivisao = SeparadorBase.criarDivisao(vetorClasses,vetorGrupos,42)
        segundaDivisao = SeparadorBase.criarDivisao(vetorClasses,vetorGrupos,42)

        self.assertIsNotNone(primeiraDivisao)
        self.assertIsNotNone(segundaDivisao)
        self.assertTrue(np.array_equal(primeiraDivisao[CHAVE_INDICES_TREINO],segundaDivisao[CHAVE_INDICES_TREINO]))
        self.assertTrue(np.array_equal(primeiraDivisao[CHAVE_INDICES_VALIDACAO],segundaDivisao[CHAVE_INDICES_VALIDACAO]))
        self.assertTrue(np.array_equal(primeiraDivisao[CHAVE_INDICES_TESTE],segundaDivisao[CHAVE_INDICES_TESTE]))

    def test_rejeitarGrupoComMultiplasClasses(self):
        vetorClasses = self.dicionarioBase[CHAVE_CLASSES].copy()
        vetorGrupos = self.dicionarioBase[CHAVE_GRUPOS].copy()

        vetorGrupos[20] = vetorGrupos[0]
        divisao = SeparadorBase.criarDivisao(vetorClasses,vetorGrupos)

        self.assertIsNone(divisao)
        self.assertTrue(SeparadorBase.ultimaExecucaoDeuErro())
        self.assertIsNotNone(SeparadorBase.getUltimoErro())


if(__name__ == "__main__"):
    unittest.main()