from Bio import Entrez, SeqIO
import csv
import time

Entrez.email = "lahed97318@apklamp.com"
Entrez.tool = "SARS-CoV2_Mutation_Study"


# Busca os IDs GenBank com base em um termo
def busca_ids_ncbi(query="SARS-CoV-2[ORGN]", max_seqs=50):
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=max_seqs)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


# Caso Especial cidade vindo ('cidade')
def limpar_valor_bruto(texto):
    if not isinstance(texto, str):
        return "Desconhecido"

    # Remove parênteses e aspas simples
    texto = texto.strip("(),'\"")
    return texto


def fetch_base_sequence():
    handle = Entrez.efetch(db="nucleotide", id="NC_045512.2", rettype="fasta", retmode="text")
    record = SeqIO.read(handle, "fasta")
    SeqIO.write(record, "wuhan_reference.fasta", "fasta")
    handle.close()


# Faz download das sequências GenBank dos IDs, salva metadados
def fetch_sequences(genbank_ids, delay=0.4):
    sequences = []
    metadados = []

    for idx, genbank_id in enumerate(genbank_ids):
        try:
            print(f"Baixando sequência {idx + 1}/{len(genbank_ids)} - ID: {genbank_id}")
            handle = Entrez.efetch(db="nucleotide", id=genbank_id, rettype="gb", retmode="text")
            record = SeqIO.read(handle, "genbank")
            sequences.append(record)

            # Coleta de metadados
            meta = {
                "seq_id": record.id,
                "descricao": record.description,
                "organismo": record.annotations.get("organism", ""),
                "fonte": record.annotations.get("source", ""),
                "pais": "NO_COUNTRY", "data_coleta": "", "nota": "",
            }

            for feature in record.features:
                if feature.type == "source" or feature.type == "qualifiers":
                    if "note" in feature.qualifiers:
                        meta["nota"] = feature.qualifiers.get("note", [""])[0]

                    if "geo_loc_name" in feature.qualifiers:
                        meta["pais"] = limpar_valor_bruto(feature.qualifiers.get("geo_loc_name", [""])[0])

                    if "collection_date" in feature.qualifiers:
                        meta["data_coleta"] = feature.qualifiers.get("collection_date", [""])[0]

            metadados.append(meta)

            handle.close()
            time.sleep(delay)

        except Exception as e:
            print(f"Erro ao baixar {genbank_id}: {e}")

    return sequences, metadados


# Salva as sequências em .fasta
def salva_arquivo(sequences, output_file="sars_cov2_sequences.fasta"):
    SeqIO.write(sequences, output_file, "fasta")
    print(f"{len(sequences)} sequências salvas em '{output_file}'.")


# Salva os metadados em .csv
def salva_metadados_csv(metadados, output_file="metadados_test.csv"):
    if not metadados:
        print("Nenhum metadado a salvar.")
        return

    keys = metadados[0].keys()
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metadados)

    print(f"Metadados salvos em '{output_file}'.")


def main():
    fetch_base_sequence()

    genbank_ids = busca_ids_ncbi(query="SARS-CoV-2[ORGN] AND complete genome", max_seqs=20000)
    sequences, metadados = fetch_sequences(genbank_ids)
    salva_arquivo(sequences)
    salva_metadados_csv(metadados)


if __name__ == "__main__":
    main()
