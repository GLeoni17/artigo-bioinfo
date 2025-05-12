import os
import sys
from collections import defaultdict
import tempfile
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Parâmetros
MUTACOES_CSV = "mutacoes.csv"
METADADOS_CSV = "metadados.csv"
TEMP_DIR = "features_temp"
TOP_N = 500
FREQ_MIN = 0.1
FREQ_MAX = 0.9
OUTPUT_X = "X_features.csv"
OUTPUT_Y = "y_labels.csv"
LABEL = "pais"
CHUNKSIZE = 50_000


# Função de worker: processa um único chunk e salva resultado
def processar_chunk(args):
    chunk_path, posicoes_selecionadas, index = args
    try:
        df = pd.read_csv(chunk_path)
        df = df.dropna(subset=["seq_id", "position"])
        df["position"] = df["position"].astype(str)

        filtered = df[df["position"].isin(posicoes_selecionadas)].copy()
        if filtered.empty:
            return None

        filtered["valor"] = 1
        temp = pd.crosstab(filtered["seq_id"], filtered["position"])
        output_path = f"{TEMP_DIR}/crosstab_{index}.csv"
        temp.to_csv(output_path)
        return output_path
    except Exception as e:
        print(f"[ERRO] Falha no chunk {index}: {e}")
        return None


# Função: Selecionar top posições informativas e gerar X
def selecionar_top_posicoes():
    print(f"Iniciando leitura em blocos de '{MUTACOES_CSV}' para análise de frequência...")

    seq_ids_unicos = set()

    # 1ª passagem: contar ocorrências distintas por posição
    try:
        posicoes_selecionadas = selecionar_posicoes(seq_ids_unicos)
    except Exception as e:
        print(f"[ERRO] Falha ao processar '{MUTACOES_CSV}': {e}")
        sys.exit(1)

    print("Iniciando leitura final para construir matriz binária...")

    # 2ª passagem: construir matriz binária com crosstab
    try:
        return matriz_crosstab(posicoes_selecionadas)
    except Exception as e:
        print(f"[ERRO] Falha ao gerar matriz binária: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def selecionar_posicoes(seq_ids_unicos):
    posicao_contagem = defaultdict(set)

    for chunk in pd.read_csv(MUTACOES_CSV, chunksize=CHUNKSIZE):
        chunk = chunk.dropna(subset=["seq_id", "position"])
        chunk["position"] = chunk["position"].astype(str)

        for row in chunk.itertuples(index=False):
            posicao_contagem[row.position].add(row.seq_id)
            seq_ids_unicos.add(row.seq_id)

    total_seq = len(seq_ids_unicos)
    print(f"{total_seq} sequências únicas detectadas.")

    # Frequência relativa
    freq = {
        pos: len(seq_ids) / total_seq
        for pos, seq_ids in posicao_contagem.items()
    }
    freq_df = pd.Series(freq)
    posicoes_selecionadas = freq_df[(freq_df >= FREQ_MIN) & (freq_df <= FREQ_MAX)] \
        .sort_values(ascending=False) \
        .head(TOP_N).index
    print(f"{len(posicoes_selecionadas)} posições selecionadas.")
    return posicoes_selecionadas


def matriz_crosstab(posicoes_selecionadas):
    os.makedirs(TEMP_DIR, exist_ok=True)
    temp_chunk_dir = tempfile.mkdtemp()
    chunk_paths = []
    print("Dividindo CSV em chunks temporários...")
    for i, chunk in enumerate(pd.read_csv(MUTACOES_CSV, chunksize=CHUNKSIZE)):
        chunk_file = os.path.join(temp_chunk_dir, f"chunk_{i}.csv")
        chunk.to_csv(chunk_file, index=False)
        chunk_paths.append((chunk_file, posicoes_selecionadas, i))

    print(f"Iniciando processamento paralelo de {len(chunk_paths)} chunks...")
    with ProcessPoolExecutor() as executor:
        result_files = list(executor.map(processar_chunk, chunk_paths))
    arquivos = [f for f in result_files if f and os.path.exists(f)]
    if not arquivos:
        print("[ERRO] Nenhum arquivo válido gerado.")
        sys.exit(1)

    print("Consolidando crosstabs em memória...")
    all_parts = [pd.read_csv(f).set_index("seq_id") for f in arquivos]
    df_bin = pd.concat(all_parts, axis=0).groupby(level=0).sum()
    df_bin.columns = [f"pos_{col}" for col in df_bin.columns]
    print(f"Matriz final gerada com shape: {df_bin.shape}")
    return df_bin


# Função: Carrega y com base nos metadados
def carregar_labels():
    try:
        df_meta = pd.read_csv(METADADOS_CSV)
        df_meta = df_meta[["seq_id", LABEL]].dropna()
        df_meta = df_meta.set_index("seq_id")
        print(f"Labels disponíveis: {df_meta[LABEL].nunique()} classes.")
        return df_meta
    except Exception as e:
        print(f"[ERRO] Falha ao carregar '{METADADOS_CSV}': {e}")
        sys.exit(1)


# Função: Codifica rótulos
def codificar_labels(y_series):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_series)
    print(f"Rótulos codificados em {len(set(y_encoded))} classes.")
    return pd.DataFrame(y_encoded, columns=["classe"])


# Execução principal
def main():
    if not os.path.exists(MUTACOES_CSV) or not os.path.exists(METADADOS_CSV):
        print("[ERRO] Arquivos obrigatórios não encontrados.")
        sys.exit(1)

    df_features = selecionar_top_posicoes()
    df_labels = carregar_labels()

    # Uniao Feature x Label
    print("Unindo features com rótulos...")
    df_final = df_features.join(df_labels, how="inner")
    df_final.dropna(inplace=True)

    X = df_final.drop(columns=[LABEL])
    y = df_final[LABEL]

    y_encoded = codificar_labels(y)

    print(f"Salvando X em '{OUTPUT_X}'...")
    X.to_csv(OUTPUT_X, index=False)
    print(f"Salvando y em '{OUTPUT_Y}'...")
    y_encoded.to_csv(OUTPUT_Y, index=False)


if __name__ == "__main__":
    main()
