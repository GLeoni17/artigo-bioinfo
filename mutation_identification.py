from Bio import SeqIO
import pandas as pd

path = "mafft_alinhado_auto/"


# Função para extrair mutações (SNVs e InDels) comparando com a referência
def extrai_mutacoes(ref_seq, seq):
    mutations = []

    # Comparar base a base
    for i, (ref_base, base) in enumerate(zip(ref_seq, seq)):
        if ref_base != base:  # Se for diferente da base de referência, é uma mutação
            mutations.append((i, ref_base, base))  # Posição, base de referência, base mutada

    return mutations


# Função para processar as sequências alinhadas e gerar um DataFrame com as mutações
def processar_sequencias(ref_seq, aligned_file):
    aligned_seqs = list(SeqIO.parse(path + aligned_file, "fasta"))

    mutation_data = []

    # Para cada sequência alinhada
    for record in aligned_seqs:
        if record.id == "Wuhan-Hu-1":  # Pular a referência Wuhan-Hu-1
            continue

        mutations = extrai_mutacoes(ref_seq, str(record.seq))

        for mutation in mutations:
            mutation_data.append({
                "seq_id": record.id,
                "position": mutation[0],
                "ref_base": mutation[1],
                "mut_base": mutation[2]
            })

    # Criar um DataFrame com as mutações
    df_mutations = pd.DataFrame(mutation_data)
    return df_mutations


# Salvar as mutações em um arquivo CSV para fácil análise posterior
def salvar_mutacoes(df_mutations, output_file="mutacoes.csv"):
    df_mutations.to_csv(output_file, index=False)
    print(f"Mutações salvas em '{output_file}'.")


# Execução principal
def main():
    # Inicialize um DataFrame vazio para armazenar as mutações de todos os blocos
    df_all_mutations = pd.DataFrame()

    # Carregar a sequência de referência Wuhan-Hu-1 (sequência alinhada)
    ref_seq = str(SeqIO.read(path + "wuhan_reference_aligned.fasta", "fasta").seq)

    # Processar os 200 blocos de sequências alinhadas
    for i in range(0, 200):
        aligned_file = f"aligned_{i}.fasta"
        df_mutations = processar_sequencias(ref_seq, aligned_file)

        # Adicionar as mutações deste bloco ao DataFrame global
        df_all_mutations = pd.concat([df_all_mutations, df_mutations], ignore_index=True)

    # Exibir as primeiras mutações
    print("Primeiras mutações extraídas:")
    print(df_all_mutations.head())

    # Salvar todas as mutações extraídas em um arquivo CSV
    salvar_mutacoes(df_all_mutations)


if __name__ == "__main__":
    main()
