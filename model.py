import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Caminhos
X_CSV = "X_features.csv"
Y_CSV = "y_labels.csv"

# 1. Carrega os dados
X = pd.read_csv(X_CSV)
y = pd.read_csv(Y_CSV).values.ravel()  # vetor 1D

# Conta quantas vezes cada classe aparece
counts = pd.Series(y).value_counts()

# Filtra apenas classes com pelo menos 2 instâncias
valid_classes = counts[counts >= 2].index
mask = pd.Series(y).isin(valid_classes)

X = X[mask]
y = y[mask]

# 2. Divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Treina o modelo
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# 4. Avaliação
y_pred = clf.predict(X_test)

print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
