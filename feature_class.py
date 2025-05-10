import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Caminhos
MUTACOES_CSV = "./mutacoes.csv"
METADADOS_CSV = "./test.csv"
OUTPUT_X = "X_features.csv"
OUTPUT_Y = "y_labels.csv"

def carregar_features_binarias(path, max_samples=1000, num_features=10):
    # Lê o arquivo de mutações, garantindo que "seq_id" seja o índice
    df = pd.read_csv(path, index_col=0)
    
    # Verifica as primeiras linhas para ter certeza
    
    # Garantir que a coluna "position" seja tratada como string
    df["position"] = df["position"].astype(str)
    
    # Marca a mutação se houver alguma (valor base diferente de '-')
    df["valor"] = df["mut_base"].apply(lambda x: 1 if x != '-' else 0)
    
    # Agora, selecionamos apenas as 10 primeiras posições para cada sequência
    # GroupBy para obter as 10 primeiras posições para cada 'seq_id'
    df_top_10 = df.groupby("seq_id").apply(lambda x: x.head(num_features))
    
    print(len(df_top_10.index))
    exit()
    
    # Pivotar para formar uma matriz binária: seq_id será o índice e position será a coluna
    df_bin = df_top_10.pivot(index="seq_id", columns="position", values="valor")
    
    # Preencher NaNs com 0 (nenhuma mutação)
    df_bin = df_bin.fillna(0)
    
    # Renomeia as colunas para um formato mais claro (com prefixo 'pos_')
    df_bin.columns = [f"pos_{col}" for col in df_bin.columns]
    
    return df_bin

# Junta features com metadados (baseado no país)
def unir_features_com_metadados(df_features, path_metadados, coluna_rotulo="pais"):
    # Carrega os metadados
    df_meta = pd.read_csv(path_metadados)
    
    # Verifica as primeiras linhas
    print(df_meta.head())
    
    # Filtro para manter apenas as colunas relevantes
    df_meta = df_meta[["seq_id", coluna_rotulo]].dropna()
    df_meta = df_meta.set_index("seq_id")
    
    # Junção dos metadados com as features de mutação
    df_final = df_features.join(df_meta, how="inner")
    
    # Remove qualquer linha que tenha dados faltantes
    df_final.dropna(inplace=True)
    
    # Separa em X (features) e y (rótulos)
    X = df_final.drop(columns=[coluna_rotulo])
    y = df_final[[coluna_rotulo]]
    
    return X, y

# Codifica os rótulos em inteiros (0,1,2...)
def codificar_rotulos(y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y.values.ravel())
    return pd.DataFrame(y_encoded, columns=["classe"])

# Execução principal
def main():
    print("🔍 Carregando e transformando mutações...")
    df_features = carregar_features_binarias(MUTACOES_CSV)

    print("🔗 Unindo com metadados...")
    X, y = unir_features_com_metadados(df_features, METADADOS_CSV, coluna_rotulo="pais")

    print("🎯 Codificando rótulos...")
    y_encoded = codificar_rotulos(y)

    print("💾 Salvando resultados...")
    X.to_csv(OUTPUT_X, index=False)
    y_encoded.to_csv(OUTPUT_Y, index=False)
    print(f"✅ Features salvas em {OUTPUT_X}, rótulos salvos em {OUTPUT_Y}.")

if __name__ == "__main__":
    main()
