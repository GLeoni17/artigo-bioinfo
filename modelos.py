import os
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def gerar_relatorios(y_test, predictions, unique_classes, target_names):
    report = pd.DataFrame(
        classification_report(
            y_test["classe"],
            predictions,
            output_dict=True,
            zero_division=0,
            labels=unique_classes,
            target_names=target_names
        )
    ).transpose()
    report.loc["accuracy", ["precision", "recall", "support"]] = np.nan
    report = report.round(5).fillna("")

    metricas_globais = report.loc[["accuracy", "macro avg", "weighted avg"]]
    metricas_classes = report.drop(index=["accuracy", "macro avg", "weighted avg"])
    return report, metricas_classes, metricas_globais


def salvar_tabela_como_imagem(df, path, table_name):
    n_rows = len(df)
    n_cols = len(df.columns) + 1  # +1 para os índices
    cell_height = 0.3
    fig_height = max(3, round(n_rows * cell_height))
    fig_width = max(6, round(n_cols * 1.2))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
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

    fig.suptitle(table_name, fontsize=16, weight='bold', y=1.02)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
    plt.savefig(path, bbox_inches='tight', pad_inches=0.2)
    plt.close()


def salvar_matriz_confusao(y_test, predictions, unique_classes, target_names, model_name):
    num_classes = len(unique_classes)
    fig_width = max(6, round(num_classes * 0.6))
    fig_height = max(5, round(num_classes * 0.5))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ConfusionMatrixDisplay.from_predictions(
        y_test["classe"], predictions,
        labels=unique_classes,
        display_labels=target_names,
        cmap='Blues',
        ax=ax
    )
    ax.set_title(f"Matriz de Confusão - {model_name}")
    plt.tight_layout()
    path = f"./metrics/{model_name}_matriz-confusao.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def custom_f1_macro(estimator, X, y):
    try:
        y_pred = estimator.predict(X)
        return f1_score(y, y_pred, average='macro', zero_division=0)
    except Exception as e:
        print(f"Erro ao calcular F1 score: {e}")
        return 0.0


def salva_melhores_parametros(best_params, model_name):
    print(f"Melhores parâmetros para {model_name}: {best_params}")
    best_params_path = f"./metrics/{model_name}_melhores-parametros.csv"
    best_params_df = pd.DataFrame([best_params])
    best_params_df.to_csv(best_params_path, index=False)


def get_names_and_classes(predictions, y_test):
    unique_classes = np.union1d(np.unique(y_test["classe"]), np.unique(predictions))
    class_to_name = {}
    for classe, nome in zip(y_test["classe"], y_test["nome"]):
        class_to_name[classe] = nome
    target_names = [class_to_name.get(cls, f"Classe {cls}") for cls in unique_classes]
    return target_names, unique_classes


def aplicar_modelo(model, param_grid, X_train, y_train, X_test, y_test):
    model_name = model.__class__.__name__
    os.makedirs("./metrics", exist_ok=True)

    print(f"Iniciando GridSearchCV para {model_name}...")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=custom_f1_macro,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train["classe"] if isinstance(y_train, pd.DataFrame) else y_train)
    best_model = grid_search.best_estimator_

    # Salvar os melhores parâmetros em CSV
    salva_melhores_parametros(grid_search.best_params_, model_name)

    predictions = best_model.predict(X_test)

    target_names, unique_classes = get_names_and_classes(predictions, y_test)

    report_full, metricas_classes, metricas_globais = gerar_relatorios(y_test, predictions, unique_classes, target_names)

    # Salvar CSV e imagem das métricas por classe
    path_classes_csv = f"./metrics/{model_name}_metricas-por-classe.csv"
    metricas_classes.to_csv(path_classes_csv)
    salvar_tabela_como_imagem(metricas_classes, path_classes_csv.replace(".csv", ".png"), f"Métricas por classe - {model_name}")

    # Salvar CSV e imagem das métricas globais
    path_globais_csv = f"./metrics/{model_name}_metricas-globais.csv"
    metricas_globais.to_csv(path_globais_csv)
    salvar_tabela_como_imagem(metricas_globais, path_globais_csv.replace(".csv", ".png"), f"Métricas globais - {model_name}")

    # Salvar matriz de confusão
    salvar_matriz_confusao(y_test, predictions, unique_classes, target_names, model_name)

    return report_full, predictions, grid_search.best_params_, best_model


def aplicar_random_forest(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    report_full, predictions, best_params, best_model = aplicar_modelo(
        RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
        param_grid,
        X_train, y_train, X_test, y_test
    )

    # Salvar importâncias das features
    print("Salvando importâncias das features...")
    importancias = pd.Series(best_model.feature_importances_, index=X_train.columns)
    importancias = importancias.sort_values(ascending=False)

    path_importancias = "./metrics/RandomForestClassifier_importancias.csv"
    importancias.to_csv(path_importancias, header=["importance"])
    print(f"Importâncias salvas em: {path_importancias}")

    return report_full, predictions, best_params, best_model


def aplicar_knn(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_neighbors': [5, 10, 15, 20],
        'weights': ['uniform', 'distance']
    }
    return aplicar_modelo(
        KNeighborsClassifier(n_jobs=-1),
        param_grid,
        X_train, y_train, X_test, y_test
    )


def aplicar_svm(X_train, y_train, X_test, y_test):
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf'],
        'C': [1, 10, 50, 100],
        'degree': [2, 3, 5]  # Apenas afeta kernel='poly'
    }
    return aplicar_modelo(
        SVC(class_weight="balanced", random_state=42),
        param_grid,
        X_train, y_train, X_test, y_test
    )


# teste do funcionamento
if __name__ == '__main__':
    from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
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
