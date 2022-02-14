import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# chackpoint 1: Load the passenger data
passengers = pd.read_csv("passengers.csv")

# Clean the Data

# chackpoint 2: Update sex column to numerical
passengers["Sex"] = passengers["Sex"].map({"female":1, "male": 0})

# checkpoint 3: Fill the nan values in the age column
print(passengers['Age'].values)
passengers['Age'].fillna(value=round(passengers["Age"].mean()), inplace=True)
print(passengers)

# chackpoint 4: Create a first class column
passengers["FirstClass"] = passengers["Pclass"].apply(lambda p: 1 if p == 1 else 0)
print(passengers)

# chackpoint 5: Create a second class column
passengers["SecondClass"] = passengers["Pclass"].apply(lambda p: 1 if p == 2 else 0)
print(passengers)

# Select and Split the Data

# checkpoint 6: Select the desired features
features = passengers[["Sex", "Age", "FirstClass", "SecondClass"]]
survival = passengers["Survived"]

# checkpoint 7: Perform train, test, split
train_features, test_features, test_features, train_labels, test_labels = train_test_split(features, survival)

# Normalize the Data

# chackpoint 8: Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Create and Evaluate the Model

# checkpoint 9: Create and train the model
model = LogisticRegression()
model.fit(train_features, train_labels)

# checkpoint 10: Score the model on the train data
print(model.score(train_features, train_labels))
# outputs:
# [891 rows x 14 columns]
# 0.7979041916167665

# The model got a score of 80% accuracy rate (a prety good score), meaning that 80% of the time the model was able to predict what the answer would be from the training data. 

# checkpoint 11: Score the model on the test data
print(model.score(test_features, test_labels))
# 0.7757847533632287

# chackpoint 12: Analyze the coefficients
print(model.coef_)
# [[ 1.24466403 -0.38257845  0.97793907  0.43845355]]
# This means that Sex and FirstClass where the features that most influenced the results of the model, whereas the other two features (Age and SecondClass where not very important)

# Predict with the Model

# checkpoint 13: Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
Me = np.array([1.0,39.0,0.0,1.0])

# checkpoint 14: Combine passenger arrays
sample_passengers = np.array([Jack, Rose, Me])

# checkpoint 15: Scale the sample passenger features
sample_passengers = scaler.transform(sample_passengers)
print(sample_passengers)
# Outputs:
# [[-0.71506099 -0.74351124 -0.54505723 -0.51492865]
#  [ 1.39848211 -0.97375383  1.83466974 -0.51492865]
#  [ 1.39848211  0.71469185 -0.54505723  1.94201662]]

# checkpoint 16: Make survival predictions!
print(model.predict(sample_passengers))
# output:
# [0 1 1] which tells us that Jack won't survive, whereas the other two passengers(Rose and me) will. Sex here is a crucial feature of our model. 

print(model.predict_proba(sample_passengers))
# outputs:
# [[0.87124828 0.12875172]
# [0.04952832 0.95047168]
# [0.23752479 0.76247521]]

# The final conculsion, based on the probability of our model above, tells us that Jack has a very hagh ptobability of deing of 87% and only 12% chance of survival.
# The other two female passengers, have a 5% and 24% probability of deaf and a high probability of surviving, with 95% for Rose and 76% for me. 





