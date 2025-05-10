import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Configurações
FEATURES_CSV = "mutacoes_1k.csv"
METADADOS_CSV = "metadados_1k.csv"
TOP_N = 500
FREQ_MIN = 0.1
FREQ_MAX = 0.9
OUTPUT_X = "X_features.csv"
OUTPUT_Y = "y_labels.csv"
ROTULO = "pais"


# Selecionar top_n posições para serem pivôs nas features
def selecionar_top_posicoes(path_mutacoes, top_n=500, freq_min=0.1, freq_max=0.9):
    df = pd.read_csv(path_mutacoes)
    total_seq = df["seq_id"].nunique()

    freq = df.groupby("position")["seq_id"].nunique() / total_seq
    freq = freq[(freq >= freq_min) & (freq <= freq_max)]
    posicoes_selecionadas = freq.sort_values(ascending=False).head(top_n).index.astype(str)

    df = df[df["position"].astype(str).isin(posicoes_selecionadas)]
    df["valor"] = 1
    df["position"] = df["position"].astype(str)

    df_bin = df.pivot_table(index="seq_id", columns="position", values="valor", fill_value=0)
    df_bin.columns = [f"pos_{col}" for col in df_bin.columns]
    return df_bin


# Carrega y baseado nos metadados
def carregar_rotulos(path_metadados, rotulo_col="pais"):
    df_meta = pd.read_csv(path_metadados)
    df_meta = df_meta[["seq_id", rotulo_col]].dropna()
    df_meta = df_meta.set_index("seq_id")
    return df_meta


def codificar_rotulos(y_series):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_series)
    return pd.DataFrame(y_encoded, columns=["classe"])


def main():
    df_features = selecionar_top_posicoes(FEATURES_CSV, top_n=TOP_N, freq_min=FREQ_MIN, freq_max=FREQ_MAX)

    df_rotulos = carregar_rotulos(METADADOS_CSV, rotulo_col=ROTULO)

    # União Features e Rótulos
    df_final = df_features.join(df_rotulos, how="inner")
    df_final.dropna(inplace=True)

    X = df_final.drop(columns=[ROTULO])
    y = df_final[ROTULO]

    y_encoded = codificar_rotulos(y)

    X.to_csv(OUTPUT_X, index=False)
    y_encoded.to_csv(OUTPUT_Y, index=False)
    print(f"X salvo em {OUTPUT_X}, y salvo em {OUTPUT_Y}.")


if __name__ == "__main__":
    main()
