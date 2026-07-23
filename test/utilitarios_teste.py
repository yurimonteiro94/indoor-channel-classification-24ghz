from pathlib import Path
import zipfile

import numpy as np

from services.constantes import FREQUENCIA_FINAL_HZ
from services.constantes import FREQUENCIA_INICIAL_HZ
from services.constantes import PREFIXO_DIRETORIO_MEDICOES
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