import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

letters = datasets.load_digits()

clf = svm.SVC(gamma=0.001, C=100)

X,y = letters.data[:-10], letters.target[:-10]
clf.fit(X,y)

print(clf.predict(letters.data[:-10]))
plt.imshow(letters.images[6], interpolation='nearest')
plt.show()