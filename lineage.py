import pandas as pd
import subprocess
import os

# Caminhos de entrada/saída
fasta_input = "sars_cov2_sequences.fasta"       # arquivo com as sequências
pangolin_output = "pangolin_output.csv"
metadados_input = "metadados.csv"
metadados_saida = "metadados_com_lineage.csv"

# 1. Executar o pangolin
def rodar_pangolin(fasta_file, output_file):
    print("🧬 Executando Pangolin para classificação de variantes...")
    cmd = ["pangolin", fasta_file, "--outfile", output_file]
    subprocess.run(cmd, check=True)
    print("✅ Pangolin finalizado.")

# 2. Cruzar os resultados do pangolin com os metadados
def integrar_lineage(metadados_csv, pangolin_csv, output_csv):
    print("🔗 Integrando metadados com resultados do pangolin...")
    meta_df = pd.read_csv(metadados_csv)
    pangolin_df = pd.read_csv(pangolin_csv)

    # Renomear para garantir compatibilidade
    pangolin_df.rename(columns={"taxon": "seq_id"}, inplace=True)

    # Mesclar as tabelas
    merged = meta_df.merge(pangolin_df[["seq_id", "lineage"]], on="seq_id", how="left")

    # Salvar resultado
    merged.to_csv(output_csv, index=False)
    print(f"✅ Metadados com linhagens salvos em '{output_csv}'")

# Execução
if __name__ == "__main__":
    rodar_pangolin(fasta_input, pangolin_output)
    integrar_lineage(metadados_input, pangolin_output, metadados_saida)
