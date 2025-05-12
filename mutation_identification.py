import random
from multiprocessing import Pool, cpu_count
from Bio import SeqIO
import pandas as pd
import os

# Configurações
OUTPUT_FILE = "mutacoes.csv"
PATH = "mafft_alinhado_auto/"
TMP_DIR = "mutacoes_tmp/"
REF_SEQ = str(SeqIO.read("wuhan_reference.fasta", "fasta").seq)
MAX_OUTPUT_SIZE = 4 * 1024**3


def extrai_mutacoes(seq, ref_seq):
    mutations = []
    for i, (ref_base, base) in enumerate(zip(ref_seq, seq)):
        if ref_base != base:
            mutations.append((i, ref_base, base))
    return mutations


def processar_sequencias(aligned_file):
    filepath = os.path.join(PATH, aligned_file)
    aligned_seqs = list(SeqIO.parse(filepath, "fasta"))
    mutation_data = []

    for record in aligned_seqs:
        if record.id == "Wuhan-Hu-1":
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
        df = processar_sequencias(aligned_file)
        if not df.empty:
            os.makedirs(TMP_DIR, exist_ok=True)
            tmp_file = os.path.join(TMP_DIR, f"mutacoes_{i}.csv")
            df.to_csv(tmp_file, index=False)
            print(f"[OK] {aligned_file} processado.")
        else:
            print(f"[OK] Nenhuma mutação em {aligned_file}.")
    except Exception as e:
        print(f"[ERRO] {aligned_file}: {e}")


def consolidar_csv():
    try:
        all_files = [
            f for f in os.listdir(TMP_DIR)
            if f.endswith(".csv")
        ]
        random.shuffle(all_files)

        print(f"Consolidando arquivos em {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
            first = True
            total_size = 0

            for f_name in all_files:
                file_path = os.path.join(TMP_DIR, f_name)
                try:
                    df = pd.read_csv(file_path)

                    # Estimar tamanho do DataFrame em bytes (conservador)
                    est_size = df.memory_usage(deep=True).sum()
                    if total_size + est_size >= MAX_OUTPUT_SIZE:
                        print(f"[INTERROMPIDO] Inclusão de '{f_name}' excederia o limite de {MAX_OUTPUT_SIZE} bytes.")
                        break

                    # Escreve cabeçalho apenas na primeira vez
                    df.to_csv(out_f, header=first, index=False, mode='a')
                    out_f.flush()
                    total_size = os.path.getsize(OUTPUT_FILE)
                    first = False
                except Exception as e:
                    print(f"[ERRO] Ao processar '{f_name}': {e}")

        print(f"[FINALIZADO] Mutações salvas em '{OUTPUT_FILE}'.")

    except Exception as e:
        print(f"[FALHA GERAL] Erro ao consolidar arquivos: {e}")


def main():
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    if os.path.exists(TMP_DIR):
        for f in os.listdir(TMP_DIR):
            os.remove(os.path.join(TMP_DIR, f))
    else:
        os.makedirs(TMP_DIR)

    aligned_files = [
        f for f in os.listdir(PATH)
        if f.startswith("aligned_") and f.endswith(".fasta")
    ]
    indices = list(range(len(aligned_files)))

    with Pool(cpu_count()) as pool:
        pool.map(processar_e_salvar, indices)

    consolidar_csv()


if __name__ == "__main__":
    main()
