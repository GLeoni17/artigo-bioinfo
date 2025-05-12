import os
import sys
import shutil
import tempfile
from collections import defaultdict
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import LabelEncoder

# Parâmetros
MUTACOES_CSV = "mutacoes.csv"
METADADOS_CSV = "metadados.csv"
TEMP_DIR = "features_temp"
TOP_N = 500
FREQ_MIN = 0.2
FREQ_MAX = 0.9
OUTPUT_X = "X_features.csv"
OUTPUT_Y = "y_labels.csv"
LABEL = "pais"
CHUNKSIZE = 50_000


# Processa chunk e gera crosstab com base nas mutações para A, T, C, G
def processar_chunk(args):
    chunk_path, posicoes_selecionadas, index = args
    try:
        df = pd.read_csv(chunk_path)
        df = df.dropna(subset=["seq_id", "position", "mut_base"])
        df["position"] = df["position"].astype(str)

        df = df[df["position"].isin(posicoes_selecionadas)]
        if df.empty:
            return None

        # Expandir casos com múltiplas mutações (ex: C,T -> duas linhas)
        df["mut_base"] = df["mut_base"].str.split(",")
        df = df.explode("mut_base")

        df = df[df["mut_base"].isin(["A", "T", "C", "G"])]
        df["feature"] = df["position"] + "_" + df["mut_base"]
        df["valor"] = 1

        crosstab = pd.crosstab(df["seq_id"], df["feature"])
        output_path = os.path.join(TEMP_DIR, f"crosstab_{index}.csv")
        crosstab.to_csv(output_path)
        return output_path
    except Exception as e:
        print(f"[ERRO] Chunk {index}: {e}")
        return None


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

    freq = {pos: len(seq_ids) / total_seq for pos, seq_ids in posicao_contagem.items()}
    freq_df = pd.Series(freq)
    posicoes = freq_df[(freq_df >= FREQ_MIN) & (freq_df <= FREQ_MAX)]
    return posicoes.sort_values(ascending=False).head(TOP_N).index.tolist()


def matriz_crosstab(posicoes_selecionadas):
    os.makedirs(TEMP_DIR, exist_ok=True)
    temp_chunk_dir = tempfile.mkdtemp()

    chunk_paths = []
    for i, chunk in enumerate(pd.read_csv(MUTACOES_CSV, chunksize=CHUNKSIZE)):
        chunk_file = os.path.join(temp_chunk_dir, f"chunk_{i}.csv")
        chunk.to_csv(chunk_file, index=False)
        chunk_paths.append((chunk_file, posicoes_selecionadas, i))

    print(f"Processando {len(chunk_paths)} chunks em paralelo...")
    with ProcessPoolExecutor() as executor:
        result_files = list(executor.map(processar_chunk, chunk_paths))

    arquivos_validos = [f for f in result_files if f and os.path.exists(f)]
    if not arquivos_validos:
        print("[ERRO] Nenhum arquivo válido gerado.")
        sys.exit(1)

    print("Consolidando crosstabs...")
    dfs = [pd.read_csv(f).set_index("seq_id") for f in arquivos_validos]
    df_bin = pd.concat(dfs, axis=0).groupby(level=0).sum()
    df_bin = df_bin.fillna(0).astype(int)
    return df_bin


def carregar_labels():
    try:
        df = pd.read_csv(METADADOS_CSV)
        df = df[["seq_id", LABEL]].dropna().set_index("seq_id")
        print(f"{df[LABEL].nunique()} classes detectadas.")
        return df
    except Exception as e:
        print(f"[ERRO] Falha ao carregar '{METADADOS_CSV}': {e}")
        sys.exit(1)


def codificar_labels(y_series):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_series)
    print(f"Rótulos codificados: {len(set(y_encoded))} classes.")
    return pd.DataFrame(y_encoded, columns=["classe"])


def selecionar_top_posicoes():
    print("Selecionando posições hotspot por frequência...")
    seq_ids_unicos = set()
    posicoes = selecionar_posicoes(seq_ids_unicos)
    print(f"{len(posicoes)} posições selecionadas.")
    return matriz_crosstab(posicoes)


def main():
    if not os.path.exists(MUTACOES_CSV) or not os.path.exists(METADADOS_CSV):
        print("[ERRO] Arquivos de entrada não encontrados.")
        sys.exit(1)

    df_features = selecionar_top_posicoes()
    df_labels = carregar_labels()

    print("Combinando features com rótulos...")
    df_final = df_features.join(df_labels, how="inner").dropna()
    X = df_final.drop(columns=[LABEL])
    y = df_final[LABEL]

    y_encoded = codificar_labels(y)

    print(f"Salvando X em '{OUTPUT_X}'...")
    X.to_csv(OUTPUT_X, index=False)
    print(f"Salvando y em '{OUTPUT_Y}'...")
    y_encoded.to_csv(OUTPUT_Y, index=False)

    shutil.rmtree(TEMP_DIR)


if __name__ == "__main__":
    main()
