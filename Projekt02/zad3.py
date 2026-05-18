import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. Wczytanie danych
df = pd.read_csv("iris_big.csv")
df = df.dropna()

# 2. Wybór kolumn
cecha_x = "sepal length (cm)"
cecha_y = "sepal width (cm)"
kolumna_klasy = "target_name"

X = df[[cecha_x, cecha_y]]
y = df[kolumna_klasy]

# 3. Skalowanie danych
skaler_minmax = MinMaxScaler()
X_minmax = skaler_minmax.fit_transform(X)

skaler_zscore = StandardScaler()
X_zscore = skaler_zscore.fit_transform(X)

# 4. Zamiana na DataFrame
df_oryginalne = X.copy()
df_oryginalne[kolumna_klasy] = y.values

df_minmax = pd.DataFrame(X_minmax, columns=[cecha_x, cecha_y])
df_minmax[kolumna_klasy] = y.values

df_zscore = pd.DataFrame(X_zscore, columns=[cecha_x, cecha_y])
df_zscore[kolumna_klasy] = y.values

# 5. Funkcja do rysowania wykresu
def rysuj_wykres(ax, dane, tytul):
    for klasa in dane[kolumna_klasy].unique():
        podzbior = dane[dane[kolumna_klasy] == klasa]
        ax.scatter(podzbior[cecha_x], podzbior[cecha_y], label=klasa, s=20)

    ax.set_title(tytul)
    ax.set_xlabel("Sepal Length (cm)")
    ax.set_ylabel("Sepal Width (cm)")
    ax.legend()

# 6. Trzy wykresy obok siebie
fig, osie = plt.subplots(1, 3, figsize=(18, 5))

rysuj_wykres(osie[0], df_oryginalne, "Dane oryginalne")
rysuj_wykres(osie[1], df_zscore, "Dane standaryzowane Z-score")
rysuj_wykres(osie[2], df_minmax, "Dane znormalizowane Min-Max")

plt.tight_layout()
plt.show()

# 7. Statystyki
print("=== Dane oryginalne ===")
print(df[[cecha_x, cecha_y]].agg(["min", "max", "mean", "std"]))

print("\n=== Dane Min-Max ===")
print(df_minmax[[cecha_x, cecha_y]].agg(["min", "max", "mean", "std"]))

print("\n=== Dane Z-score ===")
print(df_zscore[[cecha_x, cecha_y]].agg(["min", "max", "mean", "std"]))
