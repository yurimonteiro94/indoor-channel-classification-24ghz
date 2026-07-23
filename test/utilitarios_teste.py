from pathlib import Path
import zipfile

import numpy as np

from services.constantes import CHAVE_AMBIENTES
from services.constantes import CHAVE_ARQUIVOS
from services.constantes import CHAVE_CLASSES
from services.constantes import CHAVE_FREQUENCIAS
from services.constantes import CHAVE_GRUPOS
from services.constantes import CHAVE_POSICOES
from services.constantes import CHAVE_REPETICOES
from services.constantes import CHAVE_S21
from services.constantes import DICIONARIO_CLASSES_POR_DIRETORIO
from services.constantes import FREQUENCIA_FINAL_HZ
from services.constantes import FREQUENCIA_INICIAL_HZ
from services.constantes import PREFIXO_DIRETORIO_MEDICOES
from services.constantes import QUANTIDADE_AMBIENTES
from services.constantes import QUANTIDADE_CANAIS_S21
from services.constantes import QUANTIDADE_PONTOS_FREQUENCIA


def criarZipSintetico(diretorioTemporario:str) -> Path | None:
    try:
        caminhoArquivoZip = Path(diretorioTemporario) / "base_teste.zip"
        vetorFrequencias = np.linspace(FREQUENCIA_INICIAL_HZ,FREQUENCIA_FINAL_HZ,QUANTIDADE_PONTOS_FREQUENCIA)

        listaDeEspecificacoes = [
            ("Corridor_rm155_7.1","Loc_0000",1,0.001),
            ("Corridor_rm155_7.1","Loc_0000",2,0.002),
            ("Lab139_7.1","Loc_0001",1,0.003),
            ("Lab139_7.1","Loc_0001",2,0.004)
        ]

        with zipfile.ZipFile(caminhoArquivoZip,"w",compression=zipfile.ZIP_DEFLATED) as arquivoZip:
            for especificacao in listaDeEspecificacoes:
                nomeAmbiente = especificacao[0]
                posicao = especificacao[1]
                repeticao = especificacao[2]
                escala = especificacao[3]

                nomeArquivo = "Medicao_" + str(repeticao) + "Ch1.csv"
                caminhoInterno = PREFIXO_DIRETORIO_MEDICOES + nomeAmbiente + "/" + posicao + "/" + nomeArquivo

                listaDeLinhas = [
                    "# Version 1.00",
                    "#",
                    "freq[Hz];re:Trc1_S11;im:Trc1_S11;re:Trc2_S21;im:Trc2_S21;"
                ]

                for indice in range(QUANTIDADE_PONTOS_FREQUENCIA):
                    frequencia = vetorFrequencias[indice]
                    parteRealS21 = escala + indice * 0.000001
                    parteImaginariaS21 = -escala + indice * 0.0000005
                    linha = "%.15E;%.15E;%.15E;%.15E;%.15E;" % (frequencia,0.0,0.0,parteRealS21,parteImaginariaS21)
                    listaDeLinhas.append(linha)

                arquivoZip.writestr(caminhoInterno,"\n".join(listaDeLinhas))

        return caminhoArquivoZip

    except Exception:
        return None


def criarBaseSinteticaExperimentos() -> dict[str,np.ndarray] | None:
    try:
        quantidadeDeGruposPorClasse = 10
        quantidadeDeRepeticoes = 2
        quantidadeDeMedicoes = QUANTIDADE_AMBIENTES * quantidadeDeGruposPorClasse * quantidadeDeRepeticoes

        vetorFrequencias = np.linspace(FREQUENCIA_INICIAL_HZ,FREQUENCIA_FINAL_HZ,QUANTIDADE_PONTOS_FREQUENCIA)
        vetorFrequenciasNormalizadas = (vetorFrequencias - FREQUENCIA_INICIAL_HZ) / (FREQUENCIA_FINAL_HZ - FREQUENCIA_INICIAL_HZ)

        matrizS21 = np.empty((quantidadeDeMedicoes,QUANTIDADE_PONTOS_FREQUENCIA,QUANTIDADE_CANAIS_S21),dtype=np.float32)
        vetorClasses = np.empty(quantidadeDeMedicoes,dtype=np.int8)
        vetorRepeticoes = np.empty(quantidadeDeMedicoes,dtype=np.int8)
        vetorGrupos = np.empty(quantidadeDeMedicoes,dtype=np.int16)

        listaDeAmbientes = []
        listaDePosicoes = []
        listaDeArquivos = []

        dicionarioDiretoriosPorClasse = {}

        for nomeDiretorio in DICIONARIO_CLASSES_POR_DIRETORIO:
            classe = DICIONARIO_CLASSES_POR_DIRETORIO[nomeDiretorio]
            dicionarioDiretoriosPorClasse[classe] = nomeDiretorio

        indiceMedicao = 0

        for classe in range(QUANTIDADE_AMBIENTES):
            nomeAmbiente = dicionarioDiretoriosPorClasse[classe]

            for indiceGrupoLocal in range(quantidadeDeGruposPorClasse):
                grupo = classe * quantidadeDeGruposPorClasse + indiceGrupoLocal
                posicao = "Loc_%02d%02d" % (0,indiceGrupoLocal)
                amplitude = 0.001 + classe * 0.0005 + indiceGrupoLocal * 0.00001

                for repeticao in range(1,quantidadeDeRepeticoes + 1):
                    fase = 2.0 * np.pi * (classe + 1) * vetorFrequenciasNormalizadas
                    fase = fase + indiceGrupoLocal * 0.05 + repeticao * 0.01

                    matrizS21[indiceMedicao,:,0] = amplitude * np.cos(fase)
                    matrizS21[indiceMedicao,:,1] = amplitude * np.sin(fase)

                    vetorClasses[indiceMedicao] = classe
                    vetorRepeticoes[indiceMedicao] = repeticao
                    vetorGrupos[indiceMedicao] = grupo

                    listaDeAmbientes.append(nomeAmbiente)
                    listaDePosicoes.append(posicao)
                    listaDeArquivos.append(nomeAmbiente + "/" + posicao + "/Medicao_" + str(repeticao) + "Ch1.csv")

                    indiceMedicao = indiceMedicao + 1

        return {
            CHAVE_FREQUENCIAS: vetorFrequencias,
            CHAVE_S21: matrizS21,
            CHAVE_CLASSES: vetorClasses,
            CHAVE_AMBIENTES: np.asarray(listaDeAmbientes,dtype=np.str_),
            CHAVE_POSICOES: np.asarray(listaDePosicoes,dtype=np.str_),
            CHAVE_REPETICOES: vetorRepeticoes,
            CHAVE_GRUPOS: vetorGrupos,
            CHAVE_ARQUIVOS: np.asarray(listaDeArquivos,dtype=np.str_)
        }

    except Exception:
        return None