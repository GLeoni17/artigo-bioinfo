import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Parâmetros
mutacoes_csv = "mutacoes_1k.csv"
metadados_csv = "metadados_1k.csv"
top_n = 500
freq_min = 0.1
freq_max = 0.9
output_x = "X_features.csv"
output_y = "y_labels.csv"
label = "pais"

# Função: Selecionar top posições informativas e gerar X
def selecionar_top_posicoes():
    df = pd.read_csv(mutacoes_csv)
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

# Função: Carrega y com base nos metadados
def carregar_labels():
    df_meta = pd.read_csv(metadados_csv)
    df_meta = df_meta[["seq_id", label]].dropna()
    df_meta = df_meta.set_index("seq_id")
    return df_meta

# Função: Codifica rótulos
def codificar_labels(y_series):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_series)
    return pd.DataFrame(y_encoded, columns=["classe"])

# Execução principal
def main():
    df_features = selecionar_top_posicoes()

    df_labels = carregar_labels()

    # Uniao Feature x Label
    df_final = df_features.join(df_labels, how="inner")
    df_final.dropna(inplace=True)

    X = df_final.drop(columns=[label])
    y = df_final[label]

    y_encoded = codificar_labels(y)

    X.to_csv(output_x, index=False)
    y_encoded.to_csv(output_y, index=False)

main()