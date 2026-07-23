import tempfile
import unittest
from pathlib import Path

import numpy as np

from services.analisador_exploratorio import CHAVE_CONTAGENS_GRUPOS
from services.analisador_exploratorio import CHAVE_CONTAGENS_MEDICOES
from services.analisador_exploratorio import CHAVE_COORDENADAS_PCA
from services.analisador_exploratorio import CHAVE_MAGNITUDE_MEDIA_DB
from services.analisador_exploratorio import CHAVE_RESUMO_ESTABILIDADE
from services.analisador_exploratorio import CHAVE_TEMPORAL_MEDIA_DB
from services.analisador_exploratorio import CHAVE_VARIANCIA_PCA
from services.analisador_exploratorio import AnalisadorExploratorio
from services.constantes import QUANTIDADE_AMBIENTES
from services.constantes import QUANTIDADE_PONTOS_FREQUENCIA
from services.preparador_experimentos import PreparadorExperimentos
from test.utilitarios_teste import criarBaseSinteticaExperimentos


class TestAnalisadorExploratorio(unittest.TestCase):
    def setUp(self):
        self.dicionarioBase = criarBaseSinteticaExperimentos()
        self.dicionarioBasePreparada = PreparadorExperimentos.prepararBase(self.dicionarioBase)

    def test_gerarResultados(self):
        resultado = AnalisadorExploratorio.gerarResultados(self.dicionarioBase,self.dicionarioBasePreparada)

        self.assertIsNotNone(resultado)
        self.assertEqual(resultado[CHAVE_CONTAGENS_MEDICOES].shape,(3,QUANTIDADE_AMBIENTES))
        self.assertEqual(resultado[CHAVE_CONTAGENS_GRUPOS].shape,(3,QUANTIDADE_AMBIENTES))
        self.assertEqual(resultado[CHAVE_MAGNITUDE_MEDIA_DB].shape,(QUANTIDADE_AMBIENTES,QUANTIDADE_PONTOS_FREQUENCIA))
        self.assertEqual(resultado[CHAVE_TEMPORAL_MEDIA_DB].shape,(QUANTIDADE_AMBIENTES,QUANTIDADE_PONTOS_FREQUENCIA))
        self.assertEqual(resultado[CHAVE_COORDENADAS_PCA].shape,(56,2))
        self.assertEqual(resultado[CHAVE_VARIANCIA_PCA].shape,(2,))
        self.assertEqual(resultado[CHAVE_RESUMO_ESTABILIDADE].shape,(QUANTIDADE_AMBIENTES,5))
        self.assertTrue(np.all(np.isfinite(resultado[CHAVE_MAGNITUDE_MEDIA_DB])))
        self.assertTrue(np.all(np.isfinite(resultado[CHAVE_COORDENADAS_PCA])))

    def test_salvarResultados(self):
        with tempfile.TemporaryDirectory() as diretorioTemporario:
            diretorioResultados = Path(diretorioTemporario) / "resultados"
            diretorioGraficos = Path(diretorioTemporario) / "graficos"

            resultado = AnalisadorExploratorio.gerarResultados(self.dicionarioBase,self.dicionarioBasePreparada)
            listaDeArquivos = AnalisadorExploratorio.salvarResultados(resultado,diretorioResultados,diretorioGraficos)

            self.assertIsNotNone(listaDeArquivos)
            self.assertEqual(len(listaDeArquivos),12)

            for caminhoArquivo in listaDeArquivos:
                self.assertTrue(caminhoArquivo.exists())
                self.assertTrue(caminhoArquivo.is_file())
                self.assertGreater(caminhoArquivo.stat().st_size,0)


if(__name__ == "__main__"):
    unittest.main()