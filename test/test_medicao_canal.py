import unittest

import numpy as np

from model.entidades.medicao_canal import MedicaoCanal
from services.constantes import CODIGO_ERRO_CLASSE_INVALIDA
from services.constantes import FREQUENCIA_FINAL_HZ
from services.constantes import FREQUENCIA_INICIAL_HZ
from services.constantes import QUANTIDADE_PONTOS_FREQUENCIA


class TestMedicaoCanal(unittest.TestCase):
    def setUp(self):
        self.vetorFrequencias = np.linspace(FREQUENCIA_INICIAL_HZ,FREQUENCIA_FINAL_HZ,QUANTIDADE_PONTOS_FREQUENCIA)
        self.parteRealS21 = np.linspace(0.001,0.002,QUANTIDADE_PONTOS_FREQUENCIA)
        self.parteImaginariaS21 = np.linspace(-0.002,-0.001,QUANTIDADE_PONTOS_FREQUENCIA)

    def criarMedicaoValida(self):
        return MedicaoCanal.criar("Corridor_rm155_7.1",0,"Loc_0000",1,0,"arquivo.csv",self.vetorFrequencias,self.parteRealS21,self.parteImaginariaS21)

    def test_criarMedicaoValida(self):
        medicao = self.criarMedicaoValida()

        self.assertIsNotNone(medicao)
        self.assertTrue(medicao.validar())
        self.assertFalse(MedicaoCanal.ultimaExecucaoDeuErro())
        self.assertEqual(medicao.getClasse(),0)
        self.assertEqual(medicao.getPosicao(),"Loc_0000")
        self.assertEqual(medicao.getRepeticao(),1)
        self.assertEqual(medicao.getGrupo(),0)
        self.assertEqual(medicao.obterS21Complexo().shape,(QUANTIDADE_PONTOS_FREQUENCIA,))
        self.assertEqual(medicao.obterS21EmDoisCanais().shape,(QUANTIDADE_PONTOS_FREQUENCIA,2))

    def test_rejeitarClasseInvalida(self):
        medicao = MedicaoCanal.criar("Corridor_rm155_7.1",10,"Loc_0000",1,0,"arquivo.csv",self.vetorFrequencias,self.parteRealS21,self.parteImaginariaS21)
        erro = MedicaoCanal.getUltimoErro()

        self.assertIsNone(medicao)
        self.assertTrue(MedicaoCanal.ultimaExecucaoDeuErro())
        self.assertIsNotNone(erro)
        self.assertEqual(erro.getCodigo(),CODIGO_ERRO_CLASSE_INVALIDA)

    def test_setterRetornaBoolean(self):
        medicao = self.criarMedicaoValida()

        self.assertIsNotNone(medicao)
        self.assertTrue(medicao.setClasse(1))
        self.assertFalse(medicao.setClasse(10))
        self.assertEqual(medicao.getClasse(),1)

    def test_getterRetornaCopiaDoVetor(self):
        medicao = self.criarMedicaoValida()
        vetorRecebido = medicao.getParteRealS21()
        vetorRecebido[0] = 999.0

        self.assertNotEqual(medicao.getParteRealS21()[0],999.0)


if(__name__ == "__main__"):
    unittest.main()