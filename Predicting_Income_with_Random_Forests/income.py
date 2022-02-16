def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# Investigate The Data(1-4)

# checkpoint 1 & 2
income_data = pd.read_csv("income.csv", header=0, delimiter = ", ")

# chackpoint 3
print(income_data.iloc[0])
# the column is income

# checkpoint 4
#income_data = pd.read_csv("income.csv", header = 0, delimiter = ", ")

# Format The Data For Scikit-learn(5-7)

# checkpoint 5
labels = income_data[["income"]]

income_data["sex-int"] = income_data["sex"].apply(lambda row: 0 if row == "Male" else 1)

income_data["country-int"] = income_data["United-States"].apply(lambda row: 0 if row == "Male" else 1)

# checkpoint 6
data = [["age", "capital-gain", "capital-loss", "hours-per-week", "sex-int", "country-int"]]

# chckpoint 7
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)

# Create The Random Forest(8-11)

# checkpoint 8 
forest = RandomForestClassifier(random_state = 1)

# checkpoint 9
forest.fit(train_data, train_labels)

# checkpoint 10
# data = [["age", "capital_gain", "capital_loss", "hours-per-week"]]

# checkpoint 11
forest.score(test_data, test_labels)

# Changing Column Types

# checkpoint 12
# income_data["sex-int"] = income_data["sex"].apply(lambda row: 0 if row == "Male" else 1)

# checkpoint 13
# data = [["age", "capital_gain", "capital_loss", "hours-per-week", "sex-int"]]

# checkpoint 14
print(income_data["native-country"].value_counts())

# checkpoint 15
# income_data["country-int"] = income_data["United-States"].apply(lambda row: 0 if row == "Male" else 1)

# chackpoint 16
# data = [["age", "capital-gain", "capital-loss", "hours-per-week", "sex-int", "country-int"]]

# Explore On Your Own

# checkpoint 17
one_classifier = tree.DecisionTreeClassifier(train_data, train_labels)
one_classifier.fit(test_data, test_labes)
print(forest.feature_importances_)

