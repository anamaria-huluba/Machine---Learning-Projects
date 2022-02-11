from sklearn.datasets import load_breast_cancer
# checkpint 4
from sklearn.model_selection import train_test_split 
# checkpoint 8
from sklearn.neighbors import KNeighborsClassifier 
# checkpoint 13
import matplotlib.pyplot as plt

# Explore the data(checkpoints 1-3)

# checkpoint 1
breast_cancer_data = load_breast_cancer()

# chckpoint 2
print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)

# ckecpoint 3
print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

# Splitting the data into Training and Validation Sets(checkpoints 4-7)

# checkpoint 5 & 6
training_data, validation_data, training_labels, validation_labels = train_test_split(
  breast_cancer_data.data, 
  breast_cancer_data.target, 
  test_size = 0.2, 
  random_state = 100
  )

# checkpoint 7
print(len(training_data))
print(len(training_labels))

# Running the classifier(checkpoints 8-12)

# checkpoint 9
classifier = KNeighborsClassifier(n_neighbors = 3)

# checkpoint 10
classifier.fit(training_data, training_labels)

# chckpooint 11
print(classifier.score(validation_data, validation_labels))
 
# checkpoint 12 & 15
accuracies = []
for k in range(1, 101):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data, training_labels)
  accuracies.append(classifier.score(validation_data, validation_labels))

# Graphing the results(13-18)

# checkpoint 14
k_list = range(1,101)

# checkpoint 15
print(accuracies)

# chackpoint 16
plt.plot(k_list, accuracies)

# checkpoint 17
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()
