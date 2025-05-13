if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from modelos import aplicar_random_forest, aplicar_knn, aplicar_svm

    # Caminhos
    X_CSV = "X_features.csv"
    Y_CSV = "y_labels.csv"

    # 1. Carrega os dados
    X = pd.read_csv(X_CSV)
    y = pd.read_csv(Y_CSV)

    # Conta quantas vezes cada classe aparece
    counts = pd.Series(y["classe"]).value_counts()

    # Filtra apenas classes com pelo menos 10 instâncias
    valid_classes = counts[counts >= 10].index
    mask = pd.Series(y["classe"], index=X.index).isin(valid_classes)
    X = X[mask]
    y = y[mask]

    # 2. Divide em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Treina e avalia modelos
    aplicar_random_forest(X_train, y_train, X_test, y_test)

    aplicar_knn(X_train, y_train, X_test, y_test)

    aplicar_svm(X_train, y_train, X_test, y_test)
