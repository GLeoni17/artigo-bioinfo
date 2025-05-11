from Bio import SeqIO
import pandas as pd

# Configurações
PATH = "mafft_alinhado_auto/"
REF_SEQ = str(SeqIO.read(PATH + "wuhan_reference_aligned.fasta", "fasta").seq)


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


def main():
    df_all_mutations = pd.DataFrame()

    # Processar os 200 blocos de sequências alinhadas
    for i in range(0, 200):
        aligned_file = f"aligned_{i}.fasta"
        df_mutations = processar_sequencias(aligned_file)

        df_all_mutations = pd.concat([df_all_mutations, df_mutations], ignore_index=True)

    salvar_mutacoes(df_all_mutations)


if __name__ == "__main__":
    main()
