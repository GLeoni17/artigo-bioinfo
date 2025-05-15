import os
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score
import seaborn as sns
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
        ax=ax,
        normalize='all',
        xticks_rotation='vertical',
    )
    ax.set_title(f"Matriz de Confusão - {model_name}")
    ax.set_xlabel("Classe Verdadeira")
    ax.set_ylabel("Classe Predita")
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


def plotar_metricas_gridsearch(grid_search, model_name):
    results = pd.DataFrame(grid_search.cv_results_)
    score_col = "mean_test_score"
    param_cols = [col for col in results.columns if col.startswith("param_")]

    # Substituir None por string para evitar erros de plotagem
    for col in param_cols:
        results[col] = results[col].apply(lambda x: str(x) if x is not None else "None")

    # Tenta escolher 2 parâmetros principais
    principais = param_cols[:2] if len(param_cols) >= 2 else param_cols

    plt.figure(figsize=(10, 6))

    if len(principais) == 1:
        ax = sns.boxplot(data=results, x=principais[0], y=score_col)
    elif len(principais) >= 2:
        ax = sns.scatterplot(
            data=results,
            x=principais[0],
            y=score_col,
            hue=principais[1],
            palette="viridis",
            s=100
        )
    else:
        print("Nenhum hiperparâmetro detectado para plotagem.")
        return

    # Localizar e destacar o melhor ponto
    best_idx = results[score_col].idxmax()
    best_row = results.loc[best_idx]
    x_best = best_row[principais[0]]
    y_best = best_row[score_col]

    if len(principais) >= 2:
        ax.scatter(x_best, y_best, color='red', marker='*', s=200, edgecolor='black', label='Melhor desempenho')
        ax.annotate(f"{y_best:.5f}", (x_best, y_best), textcoords="offset points", xytext=(0, 10), ha='center',
                    fontsize=9, color='red')
    else:
        ax.scatter(x_best, y_best, color='red', marker='*', s=200, edgecolor='black', label='Melhor desempenho')
        ax.annotate(f"{y_best:.5f}", (x_best, y_best), textcoords="offset points", xytext=(0, 10), ha='center',
                    fontsize=9, color='red')

    # Título e eixos
    plt.title(f"F1-score (macro) por combinação de hiperparâmetros - {model_name}", fontsize=14)
    plt.ylabel("F1-score macro) (média de validação cruzada)")
    plt.xlabel(principais[0].replace("param_", "").capitalize())
    if len(principais) >= 2:
        plt.legend(title=principais[1].replace("param_", "").capitalize(), loc='lower right')
    else:
        plt.legend(loc='lower right')
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = f"./metrics/{model_name}_gridsearch_metricas.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de métricas do GridSearch salvo em: {path}")


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

    # Fazer um gráfico com a relação entre os hiperparâmetros e as métricas
    plotar_metricas_gridsearch(grid_search, model_name)
    best_model = grid_search.best_estimator_

    # Salvar os melhores parâmetros em CSV
    salva_melhores_parametros(grid_search.best_params_, model_name)

    predictions = best_model.predict(X_test)

    target_names, unique_classes = get_names_and_classes(predictions, y_test)

    report_full, metricas_classes, metricas_globais = gerar_relatorios(y_test, predictions, unique_classes,
                                                                       target_names)

    # Salvar CSV e imagem das métricas por classe
    path_classes_csv = f"./metrics/{model_name}_metricas-por-classe.csv"
    metricas_classes.to_csv(path_classes_csv)
    salvar_tabela_como_imagem(metricas_classes, path_classes_csv.replace(".csv", ".png"),
                              f"Métricas por classe - {model_name}")

    # Salvar CSV e imagem das métricas globais
    path_globais_csv = f"./metrics/{model_name}_metricas-globais.csv"
    metricas_globais.to_csv(path_globais_csv)
    salvar_tabela_como_imagem(metricas_globais, path_globais_csv.replace(".csv", ".png"),
                              f"Métricas globais - {model_name}")

    # Salvar matriz de confusão
    salvar_matriz_confusao(y_test, predictions, unique_classes, target_names, model_name)

    return report_full, predictions, grid_search.best_params_, best_model


def aplicar_random_forest(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 7]
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

    # Salvar uma árvore como exemplo
    print("Salvando imagem de uma árvore da floresta...")
    exemplo_arvore = best_model.estimators_[0]

    class_names = [str(cls) for cls in best_model.classes_]
    fig, ax = plt.subplots(figsize=(32, 16))  # tamanho grande para legibilidade
    plot_tree(
        exemplo_arvore,
        feature_names=X_train.columns,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=False,
        fontsize=10,
        ax=ax
    )

    path_arvore = "./metrics/RandomForestClassifier_exemplo_arvore.png"
    plt.tight_layout()
    plt.savefig(path_arvore, dpi=200)
    plt.close()
    print(f"Imagem da árvore salva em: {path_arvore}")

    return report_full, predictions, best_params, best_model


def aplicar_knn(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_neighbors': [5, 10, 15],
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
        'C': [10, 50, 100],
        'degree': [2, 3, 5]  # Apenas afeta kernel='poly'
    }
    return aplicar_modelo(
        SVC(class_weight="balanced", random_state=42),
        param_grid,
        X_train, y_train, X_test, y_test
    )
