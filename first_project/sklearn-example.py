from sklearn import datasets

# load iris data
iris=datasets.load_iris()

x=iris.data     # data
y=iris.target   # labels

# spliting the data as training and testing data
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)

# classifier is a decision tree
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

# train the classifier
my_classifier.fit(x_train, y_train)

# make predictions based on the test data
predictions = my_classifier.predict(x_test)

# check the accuracy of the predictions
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))