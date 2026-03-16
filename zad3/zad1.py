import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("iris_big.csv")

#podzial na zbior testowy (30%) i treningowy (70%), ziarno losowosci = 13
(train_set, test_set) = train_test_split(df.values, train_size=0.7,
random_state=297240)

def classify_iris(sl, sw, pl, pw):
    if pl < 2 :
        return("setosa")
    elif pl > 5.1 or pw > 1.7:
        return("virginica")
    else:
        return("versicolor")

good_predictions = 0
for row in test_set:
    if classify_iris(*row[:4]) == row[4]:
        good_predictions += 1
print(good_predictions)
print(good_predictions/len(test_set)*100, "%")