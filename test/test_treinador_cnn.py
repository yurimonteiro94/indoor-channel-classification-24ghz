import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from model.redes.modelo_cnn_1d import ConfiguracaoCNN
from services.constantes import QUANTIDADE_AMBIENTES
from services.treinador_cnn import CHAVE_ACURACIA_FINAL
from services.treinador_cnn import CHAVE_CONFIGURACAO
from services.treinador_cnn import CHAVE_DISPOSITIVO
from services.treinador_cnn import CHAVE_EPOCAS
from services.treinador_cnn import CHAVE_F1_FINAL
from services.treinador_cnn import CHAVE_MATRIZ_CONFUSAO
from services.treinador_cnn import TreinadorCNN


class TestTreinadorCNN(unittest.TestCase):
    def criarDados(self):
        geradorAleatorio = np.random.default_rng(42)

        quantidadePorClasse = 24
        quantidadePontos = 64
        quantidadeMedicoes = QUANTIDADE_AMBIENTES * quantidadePorClasse

        matrizDados = np.empty((quantidadeMedicoes,quantidadePontos,2),dtype=np.float32)
        vetorClasses = np.empty(quantidadeMedicoes,dtype=np.int8)

        listaDeIndicesTreino = []
        listaDeIndicesValidacao = []

        indiceMedicao = 0

        for classe in range(QUANTIDADE_AMBIENTES):
            for indiceLocal in range(quantidadePorClasse):
                eixo = np.linspace(0.0,2.0 * np.pi,quantidadePontos,dtype=np.float32)
                ruidoReal = geradorAleatorio.normal(0.0,0.05,quantidadePontos)
                ruidoImaginario = geradorAleatorio.normal(0.0,0.05,quantidadePontos)

                matrizDados[indiceMedicao,:,0] = np.sin((classe + 1) * eixo) + ruidoReal
                matrizDados[indiceMedicao,:,1] = np.cos((classe + 1) * eixo) + ruidoImaginario
                vetorClasses[indiceMedicao] = classe

                if(indiceLocal < 18):
                    listaDeIndicesTreino.append(indiceMedicao)

                else:
                    listaDeIndicesValidacao.append(indiceMedicao)

                indiceMedicao = indiceMedicao + 1

        indicesTreino = np.asarray(listaDeIndicesTreino,dtype=np.int64)
        indicesValidacao = np.asarray(listaDeIndicesValidacao,dtype=np.int64)

        return matrizDados,vetorClasses,indicesTreino,indicesValidacao

    def criarResumoMlp(self,diretorioResultados:Path) -> None:
        diretorioResultados.mkdir(parents=True,exist_ok=True)
        caminhoResumo = diretorioResultados / "mlp_resumo_validacao.csv"

        with open(caminhoResumo,"w",encoding="utf-8",newline="") as arquivoCsv:
            escritor = csv.writer(arquivoCsv)
            escritor.writerow(["representacao","melhor_epoca","epocas_executadas","perda_validacao","acuracia_validacao","f1_macro_validacao","tempo_segundos","quantidade_parametros"])
            escritor.writerow(["temporal",5,10,"0.300000","0.900000","0.900000","1.000000",39092])

    def test_treinarESalvar(self):
        matrizDados,vetorClasses,indicesTreino,indicesValidacao = self.criarDados()

        resultado = TreinadorCNN.treinar(matrizDados,vetorClasses,indicesTreino,indicesValidacao,"temporal",2,2,False,False)

        self.assertIsNotNone(resultado)
        self.assertGreater(resultado[CHAVE_EPOCAS].size,0)
        self.assertEqual(resultado[CHAVE_MATRIZ_CONFUSAO].shape,(QUANTIDADE_AMBIENTES,QUANTIDADE_AMBIENTES))
        self.assertGreaterEqual(resultado[CHAVE_ACURACIA_FINAL],0.0)
        self.assertLessEqual(resultado[CHAVE_ACURACIA_FINAL],1.0)
        self.assertGreaterEqual(resultado[CHAVE_F1_FINAL],0.0)
        self.assertLessEqual(resultado[CHAVE_F1_FINAL],1.0)
        self.assertEqual(resultado[CHAVE_DISPOSITIVO],"CPU")

        with tempfile.TemporaryDirectory() as diretorioTemporario:
            diretorioRaiz = Path(diretorioTemporario)
            diretorioResultados = diretorioRaiz / "resultados"
            diretorioGraficos = diretorioRaiz / "graficos"
            diretorioModelos = diretorioRaiz / "modelos"

            self.criarResumoMlp(diretorioResultados)

            listaDeArquivos = TreinadorCNN.salvarResultados(resultado,diretorioResultados,diretorioGraficos,diretorioModelos)

            self.assertIsNotNone(listaDeArquivos)
            self.assertEqual(len(listaDeArquivos),8)

            for caminhoArquivo in listaDeArquivos:
                self.assertTrue(caminhoArquivo.exists())
                self.assertTrue(caminhoArquivo.is_file())
                self.assertGreater(caminhoArquivo.stat().st_size,0)

    def test_treinarComConfiguracaoPersonalizada(self):
        matrizDados,vetorClasses,indicesTreino,indicesValidacao = self.criarDados()

        configuracao = ConfiguracaoCNN.criar(8,16,32,8,32,0.10,32,0.0005,0.00001)

        self.assertIsNotNone(configuracao)

        resultado = TreinadorCNN.treinar(matrizDados,vetorClasses,indicesTreino,indicesValidacao,"temporal",1,1,False,False,configuracao)

        self.assertIsNotNone(resultado)
        self.assertIs(resultado[CHAVE_CONFIGURACAO],configuracao)
        self.assertEqual(resultado[CHAVE_CONFIGURACAO].getCanaisPrimeiraCamada(),8)
        self.assertEqual(resultado[CHAVE_CONFIGURACAO].getCanaisSegundaCamada(),16)
        self.assertEqual(resultado[CHAVE_CONFIGURACAO].getCanaisTerceiraCamada(),32)
        self.assertEqual(resultado[CHAVE_CONFIGURACAO].getTamanhoLote(),32)
        self.assertAlmostEqual(resultado[CHAVE_CONFIGURACAO].getTaxaAprendizado(),0.0005)

    def test_rejeitarIndicesSobrepostos(self):
        matrizDados,vetorClasses,indicesTreino,indicesValidacao = self.criarDados()

        indicesValidacao[0] = indicesTreino[0]

        resultado = TreinadorCNN.treinar(matrizDados,vetorClasses,indicesTreino,indicesValidacao,"temporal",2,1,False,False)

        self.assertIsNone(resultado)
        self.assertTrue(TreinadorCNN.ultimaExecucaoDeuErro())
        self.assertIsNotNone(TreinadorCNN.getUltimoErro())


if(__name__ == "__main__"):
    unittest.main()