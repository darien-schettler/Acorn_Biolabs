from PIL import Image
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import linear_model
from sklearn import cluster

a = np.random.randn(4, 3)
b = np.random.randn(3, 2)
c = a*b

print(c.shape)

'''
print("SciKit is a tool that allows for easy analysis of datasets using machine learning\n"
      "Machine Learning (ML) is a method of employing algorithms with tunable parameters that can be automatically"
      "adjusted according to behavior you desire\n"
      "Machine learning looks at the previously seen data (training) and uses it to predict an "
      "outcome based on some data (test)\n\n"
      "ML is a subset of the study of Artificial Intelligence (AI)\n\n"
      "ML is split into...\n"
      "---SUPERVISED LEARNING---\n"
      "--> Classification\n"
      "--> Regression\n\n"
      " --UNSUPERVISED LEARNING ---\n"
      "--> Clustering")

input("\n\nPress Enter to continue...")

print("\n\nTo get started we will use this image a a cheat sheet to decide which alogrithm to use or persue!")

input("\n\nPress Enter to continue...")

# Note this will open the default image processing application
# sci_image = Image.open('SciKit.PNG')
# sci_image.show()

input("\n\nPress Enter to continue...")

print("\n\nWe will import the famous iris classification dataset into a df from a csv to illustrate this.\n")
iris_df = pd.read_csv('iris.csv')

print("These are the columns in the dataset...")
print(iris_df.columns.values)

print("\nThis is the shape of the dataframe...")
print(iris_df.shape)
print("\nIn other words, we have %d examples and %d columns" % (iris_df.shape[0], iris_df.shape[1]))

print("\n\nThere are 4 features:\n"
      "1. Sepal Length in cm\n"
      "2. Sepal Width in cm\n"
      "3. Petal Length in cm\n"
      "4. Petal Width in cm\n")

print("\nNOTE: The sepal is the green leafy bit between the petals\n")

print("\nThere are 3 classes that an iris can be classified into:\n"
      "1. Setosa --> 0\n"
      "2. Versicolor --> 1\n"
      "3. Virginica --> 2\n\n")

print("Here is a 3 flowers from each species...\n")
print(iris_df[iris_df.index == 1])
print(iris_df[iris_df.index == 70])
print(iris_df[iris_df.index == 140])

print("\nWe now move on to a basic classifier.\n"
      "A classifier is something that can be fit to data(X,y) and predict an output(T)\n\n")

print("The Steps Follow For Basic Classification {KNN: Kth Nearest Neighbours}\n\n"
      "Step 1 Load classifier module from sklearn import neighbors\n"
      "clf = neighbors.KNeighborsClassifier()\n\nStep 2 Learning/Training\n"
      "clf.fit(X_train,y_train)\n\nStep 3: Predict\ny2 = clf.predict(X_test)\n\n")

print("NOTE: Before we do this we must split our data into test and training data.\n")

print("To do this we first need to seperate our DF into two subsets, X (data) and y (labels)\n"
      "We do this by using the drop command and general assignment as follows:\n\n")

print("X = iris_df.drop(['Species'], axis=1)\n"
      "y = iris_df.Species\n\n")

X = iris_df.drop(['Species', 'Id'], axis=1)
y = iris_df.Species

print("After importing the sklearn tool we use the command...\n"
      "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25\n"
      "Where X and y are the previously created subsets and test_size is the percentage of data"
      "reserved for testing/validation\n\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

print("If we print X_train.head(5) followed by y_train.head(5) we will see what we got...\n")
print(X_train.head(5))
print(y_train.head(5))

print("\n\nHere are the corresponding shapes of the new subsets\n")
print("X_train has %d rows and %d columns" % (X_train.shape[0], X_train.shape[1]))
print("y_train has %d rows and 1 columns" % y_train.shape)
print("X_test has %d rows and %d columns" % (X_test.shape[0], X_test.shape[1]))
print("y_test has %d rows and 1 columns" % y_test.shape)

print("\nSo that checks out!\nNow we can fit our KNeighbors Classifier to the data\n"
      "This is done using the command clf.fit(X_train, y_train) after first declaring the model as...\n"
      "clf = neighbors.KNeighborsClassifier()")

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

print("\n\nNow we will print the clf.predict(X_test) output next to the y_test output")

print(
    pd.DataFrame({'Predicted': clf.predict(X_test), 'Actual': y_test, 'Correct?': (clf.predict(X_test) == y_test) * 1}))

print("\n\nWe now move onto regression\n")
print("lm = linear_model.LinearRegression()\n")

lm = linear_model.LinearRegression()

X = np.linspace(1, 20, 100).reshape(-1, 1)
y = X + np.random.normal(0, 1, 100).reshape(-1, 1)

print("We can create some X and y data using the following commands\n"
      "X = np.linspace(1,20,100).reshape(-1,1)\n"
      "y = X + np.random.normal(0,1,100).reshape(-1,1)\n")

print(X)
print(y)

lm.fit(X,y)

print("\n\nNow we will plot the data so we can get an idea of it before we fit the data\n"
      "On the plot of the data we will use the lm.fit(X,y) to predict...\n"
      "i.e. You can use the command plt.scatter(X,y)\n"
      "plt.plot(X, lm.predict(X), '-r'")

plt.scatter(X,y)
plt.plot(X,lm.predict(X),'-r')
#plt.show()

print("\nWe will now show k-means clustering\n"
      "We use the command k_means = cluster.KMeans(n_clusters=3\n"
      "Followed by k_means.fit(X)\n")

k_means = cluster.Kmeans(n_clusters=3)
k_means.fit(X)
print(k_means.labels_[::10])
'''