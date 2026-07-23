import unittest

from model.entidades.erro_sistema import ErroSistema


class TestErroSistema(unittest.TestCase):
    def test_criarErroValido(self):
        erro = ErroSistema.criar(1000,"Mensagem de teste","TestErroSistema.test_criarErroValido","Detalhe")

        self.assertIsNotNone(erro)
        self.assertTrue(erro.validar())
        self.assertEqual(erro.getCodigo(),1000)
        self.assertEqual(erro.getMensagem(),"Mensagem de teste")
        self.assertEqual(erro.getOrigem(),"TestErroSistema.test_criarErroValido")
        self.assertEqual(erro.getDetalhe(),"Detalhe")
        self.assertIsNotNone(erro.obterDescricaoCompleta())

    def test_rejeitarErroInvalido(self):
        erro = ErroSistema.criar(0,"","",None)

        self.assertIsNone(erro)

    def test_alterarErro(self):
        erro = ErroSistema.criar(1000,"Mensagem","Origem",None)

        self.assertIsNotNone(erro)
        self.assertTrue(erro.setCodigo(2000))
        self.assertTrue(erro.setMensagem("Nova mensagem"))
        self.assertTrue(erro.setOrigem("Nova origem"))
        self.assertTrue(erro.setDetalhe("Novo detalhe"))
        self.assertFalse(erro.setCodigo(0))


if(__name__ == "__main__"):
    unittest.main()