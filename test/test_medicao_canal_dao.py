import tempfile
import unittest
from pathlib import Path

from model.dao.medicao_canal_dao import MedicaoCanalDAO
from services.constantes import CODIGO_ERRO_ARQUIVO_NAO_ENCONTRADO
from services.constantes import QUANTIDADE_PONTOS_FREQUENCIA
from test.utilitarios_teste import criarZipSintetico


class TestMedicaoCanalDAO(unittest.TestCase):
    def setUp(self):
        self.diretorioTemporario = tempfile.TemporaryDirectory()
        self.caminhoArquivoZip = criarZipSintetico(self.diretorioTemporario.name)

    def tearDown(self):
        self.diretorioTemporario.cleanup()

    def test_listarCaminhosOrdenados(self):
        listaDeCaminhos = MedicaoCanalDAO.listarCaminhosMedicoes(self.caminhoArquivoZip)

        self.assertIsNotNone(listaDeCaminhos)
        self.assertEqual(len(listaDeCaminhos),4)
        self.assertIn("Corridor_rm155_7.1",listaDeCaminhos[0])
        self.assertTrue(listaDeCaminhos[0].endswith("Medicao_1Ch1.csv"))
        self.assertTrue(listaDeCaminhos[1].endswith("Medicao_2Ch1.csv"))
        self.assertIn("Lab139_7.1",listaDeCaminhos[2])

    def test_carregarMedicao(self):
        listaDeCaminhos = MedicaoCanalDAO.listarCaminhosMedicoes(self.caminhoArquivoZip)
        medicao = MedicaoCanalDAO.carregarMedicaoPorCaminho(listaDeCaminhos[0],self.caminhoArquivoZip)

        self.assertIsNotNone(medicao)
        self.assertEqual(medicao.getClasse(),0)
        self.assertEqual(medicao.getPosicao(),"Loc_0000")
        self.assertEqual(medicao.getRepeticao(),1)
        self.assertEqual(medicao.getGrupo(),0)
        self.assertEqual(medicao.getFrequencias().size,QUANTIDADE_PONTOS_FREQUENCIA)

    def test_carregarListaDeMedicoes(self):
        listaDeMedicoes = MedicaoCanalDAO.carregarMedicoes(self.caminhoArquivoZip,2)

        self.assertIsNotNone(listaDeMedicoes)
        self.assertEqual(len(listaDeMedicoes),2)
        self.assertEqual(listaDeMedicoes[0].getRepeticao(),1)
        self.assertEqual(listaDeMedicoes[1].getRepeticao(),2)

    def test_arquivoInexistenteRetornaNone(self):
        caminhoInexistente = Path(self.diretorioTemporario.name) / "inexistente.zip"
        listaDeCaminhos = MedicaoCanalDAO.listarCaminhosMedicoes(caminhoInexistente)
        erro = MedicaoCanalDAO.getUltimoErro()

        self.assertIsNone(listaDeCaminhos)
        self.assertTrue(MedicaoCanalDAO.ultimaExecucaoDeuErro())
        self.assertIsNotNone(erro)
        self.assertEqual(erro.getCodigo(),CODIGO_ERRO_ARQUIVO_NAO_ENCONTRADO)


if(__name__ == "__main__"):
    unittest.main()