from Bio import SeqIO
import os, subprocess
from concurrent.futures import ProcessPoolExecutor

# Configurações
INPUT_FASTA = "sars_cov2_sequences.fasta"
REF_FASTA = "wuhan_reference.fasta"
OUTPUT_DIR = "mafft_alinhado_auto"
BLOCK_SIZE = 100
N_THREADS = 12

ALL_SEQUENCES = list(SeqIO.parse(INPUT_FASTA, "fasta"))


def align_block_auto(block_index):
    block_file = f"{OUTPUT_DIR}/block_{block_index}.fasta"
    aligned_file = f"{OUTPUT_DIR}/aligned_{block_index}.fasta"

    start = block_index * BLOCK_SIZE
    end = min(start + BLOCK_SIZE, len(ALL_SEQUENCES))
    block_seqs = ALL_SEQUENCES[start:end]

    # Combina a referência com o bloco
    all_block_seqs = list(SeqIO.parse(REF_FASTA, "fasta")) + block_seqs
    SeqIO.write(all_block_seqs, block_file, "fasta")

    try:
        subprocess.run([
            "mafft", "--auto", "--thread", "4", block_file
        ], stdout=open(aligned_file, "w"), stderr=subprocess.DEVNULL, check=True)

        print(f"Bloco {block_index + 1} alinhado com sucesso.")
    except subprocess.CalledProcessError:
        print(f"Erro no alinhamento do bloco {block_index + 1}.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    num_blocks = (len(ALL_SEQUENCES) + BLOCK_SIZE - 1) // BLOCK_SIZE

    with ProcessPoolExecutor(max_workers=N_THREADS) as executor:
        executor.map(align_block_auto, range(num_blocks))


if __name__ == "__main__":
    main()
