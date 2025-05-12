from multiprocessing import Manager, Lock, Pool
from Bio import SeqIO
import pandas as pd
import os

# Configurações
OUTPUT_FILE = "mutacoes.csv"
PATH = "mafft_alinhado_auto/"
REF_SEQ = str(SeqIO.read("wuhan_reference.fasta", "fasta").seq)
CHUNK_SIZE = 250_000

lock = None
primeira_escrita_flag = None


def init_worker(l, flag):
    global lock, primeira_escrita_flag
    lock = l
    primeira_escrita_flag = flag


def extrai_mutacoes(seq, ref_seq):
    mutations = []

    for i, (ref_base, base) in enumerate(zip(ref_seq, seq)):
        if ref_base != base:
            mutations.append((i, ref_base, base))

    return mutations


# Função para processar as sequências alinhadas e gerar um DataFrame com as mutações
def processar_sequencias(aligned_file):
    filepath = os.path.join(PATH, aligned_file)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    aligned_seqs = list(SeqIO.parse(filepath, "fasta"))
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

    return pd.DataFrame(mutation_data)


def processar_e_salvar(i):
    aligned_file = f"aligned_{i}.fasta"

    try:
        df_mutations = processar_sequencias(aligned_file)
        if not df_mutations.empty:
            for j in range(0, len(df_mutations), CHUNK_SIZE):
                chunk = df_mutations.iloc[j:j + CHUNK_SIZE]

                with lock:
                    chunk.to_csv(
                        OUTPUT_FILE,
                        mode='a',
                        index=False,
                        header=primeira_escrita_flag[0]
                    )
                    if primeira_escrita_flag[0]:
                        primeira_escrita_flag[0] = False
            print(f"[OK] Dados de {aligned_file} gravados.")
        else:
            print(f"[OK] Nenhuma mutação em {aligned_file}.")
    except Exception as e:
        print(f"[ERRO] Falha ao processar {aligned_file}: {e}")


def main():
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    manager = Manager()
    l = Lock()
    flag = manager.list([True])

    aligned_files = sorted([
        f for f in os.listdir(PATH)
        if f.startswith("aligned_") and f.endswith(".fasta")
    ])
    num_files = len(aligned_files)

    with Pool(processes=os.cpu_count(), initializer=init_worker, initargs=(l, flag)) as pool:
        pool.map(processar_e_salvar, range(num_files))


    print(f"[FINALIZADO] Mutações salvas em '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()
