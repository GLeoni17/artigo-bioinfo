import pandas as pd
from Bio import Entrez, SeqIO
import time

Entrez.email = "lahed97318@apklamp.com"

def obter_pais_por_id(genbank_id):
    try:
        handle = Entrez.efetch(db="nucleotide", id=genbank_id, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()

        geo_loc_name, note = "NO_COUNTRY", ""

        for feature in record.features:
            if feature.type == "source" or feature.type == "qualifiers":
                if "geo_loc_name" in feature.qualifiers:
                    geo_loc_name = limpar_valor_bruto(feature.qualifiers["geo_loc_name"][0]) 

                if "note" in feature.qualifiers:
                    note = feature.qualifiers["note"][0]

                if geo_loc_name != "NO_COUNTRY" and note != "":
                    break

        return geo_loc_name, note
    except Exception as e:
        print(f"Erro ao processar {genbank_id}: {e}")
        return "Erro", "Erro"

# Caso Especial cidade vindo ('cidade')
def limpar_valor_bruto(texto):
    if not isinstance(texto, str):
        return "Desconhecido"
    
    # Remove parênteses e aspas simples
    texto = texto.strip("(),'\"")  
    return texto


def atualizar_csv_incremental(csv_entrada):
    df = pd.read_csv(csv_entrada)
    count = 1

    if "pais" not in df.columns:
        df["pais"] = ""

    if "nota" not in df.columns:
        df["nota"] = ""

    for idx, row in df.iterrows():
        seq_id = row["seq_id"]
        if pd.isna(row["pais"]) or row["pais"] == "Erro" or row["pais"] == "None": # Busca o pais, nem todos os dados tem nota extra sobre (e nem sei se sera util)
            pais, nota = obter_pais_por_id(seq_id)

            df.at[idx, "pais"] = pais
            df.at[idx, "nota"] = nota

            # Salva após cada atualização
            df.to_csv(csv_entrada, index=False)
            print(f"[{idx+1}/{len(df)}] {seq_id} → {pais}, {nota}")
            time.sleep(0.4)  # respeita a política de uso da NCBI
        else:
            print(f"[{idx+1}/{len(df)}] {seq_id} já possui dados → {row['pais']}, {row['nota']}")
            

# Uso
atualizar_csv_incremental("metadados.csv")