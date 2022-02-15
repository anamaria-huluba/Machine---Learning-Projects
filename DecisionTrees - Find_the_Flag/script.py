import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Checkpoint 1
flags = pd.read_csv("flags.csv", header = 0)
print(flags.head())

# checkpoint 2
print(flags.columns)

# checkpoint 3
# Algeria is in Europe

# Creating Your Data and Labels(4-6)

# checkpoint 4
labels = flags[["Landmass"]]

# checkpoint 5
data = flags[["Red", "Green", "Blue", "Gold", "White", "Black", "Orange"]]

# checkpoint 6
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)

# Make and Test the Model(7-9)

# checkpoint 7
tree = DecisionTreeClassifier()

# checkpoint 8
tree.fit(train_data, train_labels)

# checkpoint 9
score = tree.score(test_data, test_labels)
print(score)

# Output score:
# 0.3469387755102041

# Tuning the Model(10-)

# checkpoint 10
for i in range(1,20):
 tree = DecisionTreeClassifier(random_state=1, max_depth=i)
 tree.fit(train_data, train_labels)
 score = tree.score(test_data, test_labels)
 print(i, score)

 # Output scores:
# 1 0.3469387755102041
# 2 0.22448979591836735
# 3 0.3469387755102041
# 4 0.32653061224489793
# 5 0.3469387755102041
# 6 0.3469387755102041
# 7 0.3469387755102041
# 8 0.3469387755102041
# 9 0.3469387755102041
# 10 0.3469387755102041
# 11 0.3469387755102041
# 12 0.3469387755102041
# 13 0.3469387755102041
# 14 0.3469387755102041
# 15 0.3469387755102041
# 16 0.3469387755102041
# 17 0.3469387755102041
# 18 0.3469387755102041
# 19 0.3469387755102041

# checkpoint 11
scores = []
for i in range(1,20):
 tree = DecisionTreeClassifier(random_state=1, max_depth=i)
 tree.fit(train_data, train_labels)
 score = tree.score(test_data, test_labels)
 scores.append(score)

# checkpoint 12
plt.plot(range(1,20), scores)
plt.show()

# chackpoint 13: reaplce the data in flags with the new provided data
flags[["Red", "Green", "Blue", "Gold",
 "White", "Black", "Orange",
 "Circles",
"Crosses","Saltires","Quarters","Sunstars",
"Crescent","Triangle"]]




