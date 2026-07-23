import tempfile
import unittest
from pathlib import Path

import numpy as np

from services.constantes import QUANTIDADE_AMBIENTES
from services.treinador_mlp import CHAVE_ACURACIA_FINAL
from services.treinador_mlp import CHAVE_EPOCAS
from services.treinador_mlp import CHAVE_F1_FINAL
from services.treinador_mlp import CHAVE_MATRIZ_CONFUSAO
from services.treinador_mlp import TreinadorMLP


class TestTreinadorMLP(unittest.TestCase):
    def criarDados(self):
        geradorAleatorio = np.random.default_rng(42)

        quantidadePorClasse = 40
        quantidadePontos = 12
        quantidadeCanais = 2
        quantidadeMedicoes = QUANTIDADE_AMBIENTES * quantidadePorClasse

        matrizDados = np.empty((quantidadeMedicoes,quantidadePontos,quantidadeCanais),dtype=np.float32)
        vetorClasses = np.empty(quantidadeMedicoes,dtype=np.int8)

        listaDeIndicesTreino = []
        listaDeIndicesValidacao = []

        indiceMedicao = 0

        for classe in range(QUANTIDADE_AMBIENTES):
            for indiceLocal in range(quantidadePorClasse):
                matrizDados[indiceMedicao,:,0] = geradorAleatorio.normal(loc=classe * 2.0,scale=0.25,size=quantidadePontos)
                matrizDados[indiceMedicao,:,1] = geradorAleatorio.normal(loc=-classe * 2.0,scale=0.25,size=quantidadePontos)
                vetorClasses[indiceMedicao] = classe

                if(indiceLocal < 30):
                    listaDeIndicesTreino.append(indiceMedicao)

                else:
                    listaDeIndicesValidacao.append(indiceMedicao)

                indiceMedicao = indiceMedicao + 1

        indicesTreino = np.asarray(listaDeIndicesTreino,dtype=np.int64)
        indicesValidacao = np.asarray(listaDeIndicesValidacao,dtype=np.int64)

        return matrizDados,vetorClasses,indicesTreino,indicesValidacao

    def test_treinarESalvar(self):
        matrizDados,vetorClasses,indicesTreino,indicesValidacao = self.criarDados()

        resultado = TreinadorMLP.treinar(matrizDados,vetorClasses,indicesTreino,indicesValidacao,"cartesiana",4,2,False)

        self.assertIsNotNone(resultado)
        self.assertGreater(resultado[CHAVE_EPOCAS].size,0)
        self.assertEqual(resultado[CHAVE_MATRIZ_CONFUSAO].shape,(QUANTIDADE_AMBIENTES,QUANTIDADE_AMBIENTES))
        self.assertGreaterEqual(resultado[CHAVE_ACURACIA_FINAL],0.0)
        self.assertLessEqual(resultado[CHAVE_ACURACIA_FINAL],1.0)
        self.assertGreaterEqual(resultado[CHAVE_F1_FINAL],0.0)
        self.assertLessEqual(resultado[CHAVE_F1_FINAL],1.0)

        with tempfile.TemporaryDirectory() as diretorioTemporario:
            diretorioRaiz = Path(diretorioTemporario)
            diretorioResultados = diretorioRaiz / "resultados"
            diretorioGraficos = diretorioRaiz / "graficos"
            diretorioModelos = diretorioRaiz / "modelos"

            dicionarioResultados = {
                "cartesiana": resultado
            }

            listaDeArquivos = TreinadorMLP.salvarResultados(dicionarioResultados,diretorioResultados,diretorioGraficos,diretorioModelos)

            self.assertIsNotNone(listaDeArquivos)
            self.assertEqual(len(listaDeArquivos),7)

            for caminhoArquivo in listaDeArquivos:
                self.assertTrue(caminhoArquivo.exists())
                self.assertTrue(caminhoArquivo.is_file())
                self.assertGreater(caminhoArquivo.stat().st_size,0)

    def test_rejeitarIndicesSobrepostos(self):
        matrizDados,vetorClasses,indicesTreino,indicesValidacao = self.criarDados()

        indicesValidacao[0] = indicesTreino[0]

        resultado = TreinadorMLP.treinar(matrizDados,vetorClasses,indicesTreino,indicesValidacao,"cartesiana",2,1,False)

        self.assertIsNone(resultado)
        self.assertTrue(TreinadorMLP.ultimaExecucaoDeuErro())
        self.assertIsNotNone(TreinadorMLP.getUltimoErro())


if(__name__ == "__main__"):
    unittest.main()