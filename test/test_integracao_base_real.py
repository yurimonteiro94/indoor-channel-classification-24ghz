import unittest

from model.dao.medicao_canal_dao import MedicaoCanalDAO
from services.constantes import CAMINHO_ARQUIVO_ZIP
from services.constantes import QUANTIDADE_MEDICOES_TOTAL
from services.constantes import QUANTIDADE_PONTOS_FREQUENCIA


@unittest.skipUnless(CAMINHO_ARQUIVO_ZIP.exists(),"A base real não está disponível localmente")
class TestIntegracaoBaseReal(unittest.TestCase):
    def test_quantidadeEPrimeiraMedicao(self):
        quantidadeDeMedicoes = MedicaoCanalDAO.contarMedicoes()
        listaDeCaminhos = MedicaoCanalDAO.listarCaminhosMedicoes()

        self.assertEqual(quantidadeDeMedicoes,QUANTIDADE_MEDICOES_TOTAL)
        self.assertIsNotNone(listaDeCaminhos)

        medicao = MedicaoCanalDAO.carregarMedicaoPorCaminho(listaDeCaminhos[0])

        self.assertIsNotNone(medicao)
        self.assertEqual(medicao.getNomeAmbiente(),"Corridor_rm155_7.1")
        self.assertEqual(medicao.getPosicao(),"Loc_0000")
        self.assertEqual(medicao.getRepeticao(),1)
        self.assertEqual(medicao.getGrupo(),0)
        self.assertEqual(medicao.getFrequencias().size,QUANTIDADE_PONTOS_FREQUENCIA)


if(__name__ == "__main__"):
    unittest.main()