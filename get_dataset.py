from Bio import Entrez, SeqIO
import csv, time

# Configurações
Entrez.email = "lahed97318@apklamp.com"
Entrez.tool = "SARS-CoV2_Mutation_Study"


# Retirar caracteres especiais
def limpar_valor_bruto(texto):
    return texto.strip("(),'\"")


# Salva as sequências em .fasta
def salva_arquivo(sequences):
    output_file = "sars_cov2_sequences.fasta"
    SeqIO.write(sequences, output_file, "fasta")


# Salva os metadados em .csv
def salva_metadados_csv(metadados, ):
    output_file = "metadados.csv"
    keys = metadados[0].keys()

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metadados)


def busca_ids_ncbi(query, max_seqs):
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=max_seqs)
    record = Entrez.read(handle)  # Apenas IDs
    handle.close()
    return record["IdList"]


def get_base_sequence():
    handle = Entrez.efetch(db="nucleotide", id="NC_045512.2", rettype="fasta", retmode="text")
    record = SeqIO.read(handle, "fasta")
    SeqIO.write(record, "wuhan_reference.fasta", "fasta")
    handle.close()


# Faz download das sequências GenBank dos IDs + salva metadados
def fetch_sequences(genbank_ids):
    sequences, metadados = [], []
    delay = 0.4

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


def main():
    get_base_sequence()
    genbank_ids = busca_ids_ncbi(query="SARS-CoV-2[ORGN] AND complete genome", max_seqs=20000)
    sequences, metadados = fetch_sequences(genbank_ids)
    salva_arquivo(sequences)
    salva_metadados_csv(metadados)


if __name__ == "__main__":
    main()
