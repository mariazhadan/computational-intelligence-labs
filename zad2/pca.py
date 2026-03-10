import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Wczytanie danych
df = pd.read_csv("iris_big.csv")

# Usuwamy wiersze z brakującymi danymi
df = df.dropna()

# 2. Kolumna docelowa
y = df["target_name"]

# 3. Cechy do PCA
X = df.drop(columns=["target_name"])
X = X.select_dtypes(include=["number"])

print("Cechy wejściowe:")
print(X.head())

print("\nEtykiety klas:")
print(y.head())

# 4. Standaryzacja danych
skaler = StandardScaler()
X_standaryzowane = skaler.fit_transform(X)

# 5. PCA dla wszystkich składowych
pca = PCA()
X_pca = pca.fit_transform(X_standaryzowane)

# 6. Nowe nazwy kolumn
kolumny_pca = [f"SK{i+1}" for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(X_pca, columns=kolumny_pca)
df_pca["target_name"] = y.values

print("\nDane po PCA:")
print(df_pca.head())

# 7. Wariancja wyjaśniona
wyjasniona_wariancja = pca.explained_variance_ratio_
skumulowana_wariancja = np.cumsum(wyjasniona_wariancja)

print("\nUdział wariancji wyjaśnionej przez kolejne składowe:")
for i, var in enumerate(wyjasniona_wariancja, start=1):
    print(f"SK{i}: {var:.4f}")

print("\nSkumulowana wariancja:")
for i, var in enumerate(skumulowana_wariancja, start=1):
    print(f"SK1..SK{i}: {var:.4f}")

# 8. Ile składowych zostawić, aby zachować minimum 95% wariancji
liczba_zostawionych = np.argmax(skumulowana_wariancja >= 0.95) + 1
liczba_usunietych = X.shape[1] - liczba_zostawionych

# Strata informacji = suma wariancji usuniętych składowych
strata_informacji = wyjasniona_wariancja[liczba_zostawionych:].sum()

print(f"\nLiczba pozostawionych składowych: {liczba_zostawionych}")
print(f"Liczba usuniętych składowych: {liczba_usunietych}")
print(f"Strata informacji: {strata_informacji:.4f} ({strata_informacji * 100:.2f}%)")

# 9. Zmniejszony zbiór danych
wybrane_kolumny = [f"SK{i+1}" for i in range(liczba_zostawionych)]
df_zredukowany = df_pca[wybrane_kolumny].copy()
df_zredukowany["target_name"] = y.values

print("\nZredukowany zbiór danych:")
print(df_zredukowany.head())

# 10. Wykres punktowy
if liczba_zostawionych == 2:
    plt.figure(figsize=(8, 6))
    for klasa in df_zredukowany["target_name"].unique():
        podzbior = df_zredukowany[df_zredukowany["target_name"] == klasa]
        plt.scatter(podzbior["SK1"], podzbior["SK2"], label=klasa)

    plt.xlabel("Składowa główna 1")
    plt.ylabel("Składowa główna 2")
    plt.title("PCA dla zbioru iris_big.csv")
    plt.legend(title="Klasa")
    plt.grid(True)
    plt.show()

elif liczba_zostawionych >= 3:
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for klasa in df_zredukowany["target_name"].unique():
        podzbior = df_zredukowany[df_zredukowany["target_name"] == klasa]
        ax.scatter(podzbior["SK1"], podzbior["SK2"], podzbior["SK3"], label=klasa)

    ax.set_xlabel("Składowa główna 1")
    ax.set_ylabel("Składowa główna 2")
    ax.set_zlabel("Składowa główna 3")
    ax.set_title("PCA dla zbioru iris_big.csv")
    ax.legend(title="Klasa")
    plt.show()



# Skumulowana wariancja:
# SK1..SK1: 0.7422
# SK1..SK2: 0.9546
# SK1..SK3: 0.9929
# SK1..SK4: 1.0000
# 
# 
# Liczba pozostawionych składowych: 2
# Liczba usuniętych składowych: 2
# Strata informacji: 0.0454 (4.54%)
# 