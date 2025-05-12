from multiprocessing import Manager, Lock
from Bio import SeqIO
import pandas as pd
from joblib import Parallel, delayed
import os

# Configurações
OUTPUT_FILE = "mutacoes.csv"
PATH = "mafft_alinhado_auto/"
REF_SEQ = str(SeqIO.read("wuhan_reference.fasta", "fasta").seq)
CHUNK_SIZE = 250_000


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


def processar_e_salvar(i, lock, primeira_escrita):
    aligned_file = f"aligned_{i}.fasta"

    try:
        df = processar_sequencias(aligned_file)
        if df.empty:
            return

        for j in range(0, len(df), CHUNK_SIZE):
            chunk = df.iloc[j:j + CHUNK_SIZE]

            with lock:
                header = False
                if primeira_escrita["value"]:
                    header = True
                    primeira_escrita["value"] = False

                chunk.to_csv(OUTPUT_FILE, mode='a', index=False, header=header)

        print(f"[OK] {aligned_file} processado e salvo.")
    except Exception as e:
        print(f"[ERRO] Falha ao processar {aligned_file}: {e}")


def main():
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    manager = Manager()
    lock = Lock()
    primeira_escrita = manager.dict()
    primeira_escrita["value"] = True

    arquivos_indices = [
        int(f.split('_')[1].split('.')[0])  # extrai índice do nome
        for f in os.listdir(PATH)
        if f.startswith("aligned_") and f.endswith(".fasta")
    ]
    max_idx = max(arquivos_indices, default=-1)
    if max_idx < 0:
        print("Nenhum arquivo 'aligned_*.fasta' encontrado.")
        return

    Parallel(n_jobs=-2)(
        delayed(processar_e_salvar)(i, lock, primeira_escrita)
        for i in range(max_idx + 1)
    )

    print(f"Mutações salvas em '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()
