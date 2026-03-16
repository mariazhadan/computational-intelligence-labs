import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


df = pd.read_csv("diagnosis.csv")

# a) Trójwymiarowy wykres punktowy
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

healthy = df[df['diagnosis'] == 0]
sick = df[df['diagnosis'] == 1]

ax.scatter(healthy['param1'], healthy['param2'], healthy['param3'], c='blue', label='Zdrowy (0)', alpha=0.6)
ax.scatter(sick['param1'], sick['param2'], sick['param3'], c='red', label='Chory (1)', alpha=0.8)

ax.set_xlabel('Parametr 1')
ax.set_ylabel('Parametr 2')
ax.set_zlabel('Parametr 3')
ax.set_title('Wykres 3D parametrów medycznych')
ax.legend()
plt.show()

# Przygotowanie do klasyfikacji
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Drzewo Decyzyjne": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB()
}

# b) Obliczanie miar i rysowanie macierzy Seabornem
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}")
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Zdrowy', 'Chory'], yticklabels=['Zdrowy', 'Chory'])
    plt.title(f"Macierz błędów: {name}")
    plt.ylabel('Prawda')
    plt.xlabel('Predykcja')
    plt.show()