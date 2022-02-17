import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans 

# Getting Started with the Digits Dataset(1-5):

# checkpoint 1
from sklearn import datasets

digits = datasets.load_digits()
print(digits)

# checkpoint 2
print(digits.DESCR)

# checkpoint 3
print(digits.data)

# checkpint 4
print(digits.target)

# checkpoint 5
print(digits.target[100])

plt.pink() 
 
plt.matshow(digits.images[100])
 
plt.show()

print(digits.target[100])

# How you can visualize more than one image:
# Figure size (width, height)
 
fig = plt.figure(figsize=(6, 6))
 
# Adjust the subplots 
 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
 
# For each of the 64 images
 
for i in range(64):
 
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
 
    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
 
    # Display an image at the i-th position
 
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
 
    # Label the image with the target value
 
    ax.text(0, 7, str(digits.target[i]))
 
plt.show()

# K-Means Clustering(6-8):

# checkpoint 6 
#Import KMeans from sklearn.cluster

# checkpoint 7
# Because there are 10 digits (0, 1, 2, 3, 4, 5, 6, 7, 8, and 9), there should be 10 clusters. So k, the number of clusters, is 10:
model = KMeans(n_clusters=10, random_state=42)

# checkpoint 8
model.fit(digits.data)

# Visualizing after K-Means(9-12):

# checkpoint 9
fig = plt.figure(figsize=(8, 3))
 
fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

# checkpoint 10
for i in range(10):
 
  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)
 
  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

# checkpoint 11
plt.show()

# checkpoint 12

# Testing Your Model:

# checkpoint 13 & 14 is getting the array of the number we are tring to feed to the model 

# checkpoint 15 I used number 2051 and produced an array:
new_samples = np.array([
[0.00,2.06,6.94,7.62,7.62,7.09,0.69,0.00,0.23,7.02,6.48,2.21,1.98,7.62,2.90,0.00,0.76,7.62,2.59,0.00,1.30,7.62,2.82,0.00,0.00,2.06,0.23,0.08,6.10,7.40,0.92,0.00,0.00,0.00,0.00,4.35,7.63,2.59,0.00,0.00,0.00,0.00,3.74,7.62,4.50,0.00,0.00,0.00,0.00,3.97,7.62,7.55,3.73,2.06,0.84,0.00,0.00,4.43,6.10,6.49,7.63,7.62,7.55,6.86],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.98,7.55,7.62,7.55,4.27,0.08,0.00,0.00,4.95,7.47,3.28,5.03,7.62,4.42,0.00,0.00,6.10,5.11,0.00,0.00,4.73,7.62,1.68,0.00,6.10,4.57,0.00,0.00,0.61,7.40,4.50,0.00,5.87,6.33,0.00,0.00,0.00,5.64,5.34,0.00,3.20,7.62,3.20,0.00,0.23,6.33,5.34,0.00,0.15,6.02,7.62,7.62,7.62,7.62,2.67],
[0.00,0.00,4.12,6.78,7.62,4.35,0.00,0.00,0.00,0.38,7.55,5.80,2.90,0.92,0.00,0.00,0.00,1.60,7.62,4.73,3.05,2.06,0.00,0.00,0.00,2.29,7.62,7.62,7.62,7.62,5.64,0.69,0.00,0.92,4.35,0.91,0.23,3.97,7.62,4.12,0.00,0.00,0.00,0.00,0.00,0.00,5.64,5.34,0.00,0.08,1.53,1.52,1.52,3.05,7.17,5.26,0.00,2.14,7.62,7.62,7.62,7.62,5.95,0.92],
[0.00,0.00,0.15,6.71,7.02,0.00,0.00,0.00,0.00,0.00,3.43,7.62,7.62,1.45,0.00,0.00,0.00,0.00,6.63,6.10,7.62,2.74,0.00,0.00,0.00,0.00,0.91,0.46,7.40,3.74,0.00,0.00,0.00,0.00,0.00,0.00,6.79,4.65,0.00,0.00,0.00,0.00,0.00,0.00,5.57,5.41,0.00,0.00,0.00,0.00,0.00,0.00,4.96,6.10,0.00,0.00,0.00,0.00,0.00,0.00,3.21,4.43,0.00,0.00]
])

# checkpoint 16
new_labels = model.predict(new_samples)

# checkpoint 17
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')
  
# output number: 8465

# checkpoint 18
# The model in not recognising my handwriting, as the data fed to it is not big enough. 



