from Bio import SeqIO
import pandas as pd
from joblib import Parallel, delayed

# Configurações
PATH = "mafft_alinhado_auto/"
REF_SEQ = str(SeqIO.read("wuhan_reference.fasta", "fasta").seq)


def extrai_mutacoes(seq, ref_seq):
    mutations = []

    for i, (ref_base, base) in enumerate(zip(ref_seq, seq)):
        if ref_base != base:
            mutations.append((i, ref_base, base))

    return mutations


def salvar_mutacoes(df_mutations):
    output_file = "mutacoes.csv"
    df_mutations.to_csv(output_file, index=False)
    print(f"Mutações salvas em '{output_file}'.")


# Função para processar as sequências alinhadas e gerar um DataFrame com as mutações
def processar_sequencias(aligned_file):
    aligned_seqs = list(SeqIO.parse(PATH + aligned_file, "fasta"))

    mutation_data = []

    for record in aligned_seqs:
        if record.id == "Wuhan-Hu-1":  # Pular a referência Wuhan-Hu-1
            continue

        mutations = extrai_mutacoes(str(record.seq), REF_SEQ)

        for mutation in mutations:
            mutation_data.append({
                "seq_id": record.id,
                "position": mutation[0],
                "ref_base": mutation[1],
                "mut_base": mutation[2]
            })

    df_mutations = pd.DataFrame(mutation_data)
    return df_mutations


def processar_e_retornar(i):
    aligned_file = f"aligned_{i}.fasta"
    mutations = processar_sequencias(aligned_file)
    print(f"Mutações da sequência {i} processadas.")
    return mutations


def main():
    resultados = Parallel(n_jobs=-2)(delayed(processar_e_retornar)(i) for i in range(200))
    df_all_mutations = pd.concat(resultados, ignore_index=True)
    salvar_mutacoes(df_all_mutations)


if __name__ == "__main__":
    main()
