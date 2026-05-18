import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Wczytanie danych
df = pd.read_csv("iris_big.csv")
X = df.iloc[:, :4]  # cechy (sl, sw, pl, pw)
y = df.iloc[:, 4]   # gatunek

# a) Podział na zbiór treningowy (70%) i testowy (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=297240
)

# b) Definicja klasyfikatorów
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "k-NN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "k-NN (k=11)": KNeighborsClassifier(n_neighbors=11),
    "Naive Bayes": GaussianNB(),
    "MLP (Sieć neuronowa)": MLPClassifier(max_iter=2000, random_state=1)
}

results = {}

print("--- EWALUACJA KLASYFIKATORÓW ---")

for name, clf in models.items():
    # Trenowanie
    clf.fit(X_train, y_train)
    # Przewidywanie
    y_pred = clf.predict(X_test)
    
    # Obliczanie dokładności
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f"\nModel: {name}")
    print(f"Dokładność (Accuracy): {acc*100:.2f}%")
    print("Macierz błędu (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))

# c) Porównanie i wyłonienie najlepszego
print("\n" + "="*30)
print("RANKING DOKŁADNOŚCI:")
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for name, acc in sorted_results:
    print(f"{name}: {acc*100:.2f}%")

print(f"\nNajlepszy klasyfikator to: {sorted_results[0][0]}")