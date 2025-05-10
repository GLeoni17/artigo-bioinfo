import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def gerar_relatorios(y_test, predictions):
    report = pd.DataFrame(classification_report(y_test, predictions, output_dict=True, zero_division=0)).transpose()
    report.loc["accuracy", ["precision", "recall", "support"]] = np.nan
    report = report.round(5).fillna("")

    metricas_globais = report.loc[["accuracy", "macro avg", "weighted avg"]]
    metricas_classes = report.drop(index=["accuracy", "macro avg", "weighted avg"])
    return report, metricas_classes, metricas_globais


def salvar_tabela_como_imagem(df, path):
    fig, ax = plt.subplots()
    ax.axis("off")

    table_data = [[idx] + list(row) for idx, row in df.iterrows()]
    column_labels = [""] + df.columns.tolist()
    table = ax.table(cellText=table_data, colLabels=column_labels, cellLoc='center', loc='center')

    table.auto_set_font_size(True)
    table.scale(1.5, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        elif col == 0:
            cell.set_text_props(weight='bold')
            label = cell.get_text().get_text()
            if label in ["accuracy", "macro avg", "weighted avg"]:
                cell.set_facecolor('#bac1f2')
                cell.set_text_props(style='italic', color='black')
                for c in range(1, len(column_labels)):
                    tcell = table[(row, c)]
                    tcell.set_facecolor('#f9f9f9')
                    tcell.set_text_props(style='italic')

    for col_idx in range(len(column_labels)):
        table.auto_set_column_width(col=col_idx)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.savefig(path, bbox_inches='tight', pad_inches=0.0)
    plt.close()


def salvar_matriz_confusao(y_test, predictions, y_labels, model_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions,
        labels=np.unique(y_labels),
        cmap='Blues',
        ax=ax
    )
    ax.set_title(f"Matriz de Confusão - {model_name}")
    plt.tight_layout()
    path = f"./metrics/{model_name}-matriz_confusao.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def aplicar_modelo(model, X_train, y_train, X_test, y_test):
    model_name = model.__class__.__name__
    os.makedirs("./metrics", exist_ok=True)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Gerar relatórios
    report_full, metricas_classes, metricas_globais = gerar_relatorios(y_test, predictions)

    # Salvar CSV e imagem das métricas por classe
    path_classes_csv = f"./metrics/{model_name}_metricas_por_classe.csv"
    metricas_classes.to_csv(path_classes_csv)
    salvar_tabela_como_imagem(metricas_classes, path_classes_csv.replace(".csv", ".png"))

    # Salvar CSV e imagem das métricas globais
    path_globais_csv = f"./metrics/{model_name}_metricas_globais.csv"
    metricas_globais.to_csv(path_globais_csv)
    salvar_tabela_como_imagem(metricas_globais, path_globais_csv.replace(".csv", ".png"))

    # Salvar matriz de confusão
    salvar_matriz_confusao(y_test, predictions, y_train, model_name)

    return report_full, predictions


def aplicar_random_forest(X_train, y_train, X_test, y_test, n_estimators=500):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    return aplicar_modelo(model, X_train, y_train, X_test, y_test)


def aplicar_knn(X_train, y_train, X_test, y_test, n_neighbors=10):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    return aplicar_modelo(model, X_train, y_train, X_test, y_test)


def aplicar_svm(X_train, y_train, X_test, y_test, kernel='poly'):
    model = SVC(kernel=kernel)
    return aplicar_modelo(model, X_train, y_train, X_test, y_test)


# teste do funcionamento
if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris

    ds_iris = load_iris()

    df_iris = pd.DataFrame(ds_iris.data, columns=ds_iris.feature_names)

    df_iris['target'] = [ds_iris.target_names[i] for i in ds_iris.target]

    # Separar features (X) e rótulos (y)
    X = df_iris.drop(columns='target')
    y = df_iris['target']

    # Dividir em treino e teste, estratificando pelas classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    aplicar_random_forest(X_train, y_train, X_test, y_test)

    aplicar_knn(X_train, y_train, X_test, y_test)

    aplicar_svm(X_train, y_train, X_test, y_test)
