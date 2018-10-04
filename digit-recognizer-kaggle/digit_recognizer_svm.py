import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

labeled_images = pd.read_csv('train.csv')
images = labeled_images.iloc[0:, 1:]
labels = labeled_images.iloc[0:, :1]
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

# Grayscale images as it is
i = 1
'''img = train_images.iloc[i].as_matrix().reshape((28, 28))
plt.imshow(img, cmap='gray')
plt.title(train_labels.iloc[i, 0])
plt.show()

plt.hist(train_images.iloc[i])
plt.show()

clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
score = clf.score(test_images, test_labels)
print('Score with grayscale images: ' + str(score))'''

# Binary images
print(test_images)
test_images[test_images > 0] = 1
train_images[train_images > 0] = 1

img = train_images.iloc[i].as_matrix().reshape((28, 28))
plt.imshow(img, cmap='binary')
plt.title(train_labels.iloc[i, 0])
plt.show()

clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
score = clf.score(test_images, test_labels)
print('Score with binary images: ' + str(score))

test_data = pd.read_csv('test.csv')
test_data[test_data > 0] = 1
results = clf.predict(test_data[0:])

print(results)

df = pd.DataFrame(results)
df.index.name = 'ImageId'
df.index += 1
df.columns = ['Label']
df.to_csv('results.csv', header=True)
