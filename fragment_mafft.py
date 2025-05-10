import os
import subprocess
from Bio import SeqIO
from concurrent.futures import ProcessPoolExecutor

# Configurações
input_fasta = "sars_cov2_sequences.fasta"
ref_fasta = "wuhan_reference.fasta"  # Sequência base
output_dir = "mafft_alinhado_auto"
block_size = 100  # Tamanho do bloco
n_threads = 12    # Número de processos paralelos (ajuste conforme seu CPU)

os.makedirs(output_dir, exist_ok=True)

# Carrega todas as sequências
all_sequences = list(SeqIO.parse(input_fasta, "fasta"))
print(f"Total de sequências carregadas: {len(all_sequences)}")

# Define função para processar um bloco de alinhamento com MAFFT --auto
def align_block_auto(block_index):
    block_file = f"{output_dir}/block_{block_index}.fasta"
    aligned_file = f"{output_dir}/aligned_{block_index}.fasta"

    start = block_index * block_size
    end = min(start + block_size, len(all_sequences))
    block_seqs = all_sequences[start:end]

    # Combina a referência com o bloco
    all_block_seqs = list(SeqIO.parse(ref_fasta, "fasta")) + block_seqs
    SeqIO.write(all_block_seqs, block_file, "fasta")

    try:
        subprocess.run([
            "mafft", "--auto", "--thread", "4", block_file
        ], stdout=open(aligned_file, "w"), stderr=subprocess.DEVNULL, check=True)

        print(f"✅ Bloco {block_index + 1} alinhado com sucesso.")
    except subprocess.CalledProcessError:
        print(f"❌ Erro no alinhamento do bloco {block_index + 1}.")

# Dispara os processos paralelos
def main():
    num_blocks = (len(all_sequences) + block_size - 1) // block_size
    print(f"Iniciando alinhamento paralelo em {num_blocks} blocos...\n")

    with ProcessPoolExecutor(max_workers=n_threads) as executor:
        executor.map(align_block_auto, range(num_blocks))

    print("\n✅ Alinhamento finalizado.")

if __name__ == "__main__":
    main()
