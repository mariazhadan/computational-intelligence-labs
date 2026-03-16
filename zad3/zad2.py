from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("iris_big.csv")

X = df.iloc[:, :4] 
y = df.iloc[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=297240)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

plt.figure(figsize=(35,35))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=clf.classes_)
plt.show()

acc = clf.score(X_test, y_test)
print(f"Score: {acc * 100:.2f}%")

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Macierz błedów",cm)