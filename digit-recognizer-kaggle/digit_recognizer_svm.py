import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

labeled_images = pd.read_csv('train.csv')
images = labeled_images.iloc[0:5000, 1:]
labels = labeled_images.iloc[0:5000, :1]
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

# Grayscale images as it is
i = 1
img = train_images.iloc[i].as_matrix()
img = img.reshape((28, 28))
plt.imshow(img, cmap='gray')
plt.title(train_labels.iloc[i, 0])
plt.show()

plt.hist(train_images.iloc[i])
plt.show()

clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
score = clf.score(test_images, test_labels)
print(score)
