from sklearn import datasets
import random
from scipy.spatial import distance

# def euc method to calculate point distance
def euc(a, b):
    return distance.euclidean(a, b)

# classifier class
class ScrappyKNN():
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predictions = []
        for row in x_test:
            label=self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_distance=euc(row, self.x_train[0])
        best_index=0
        for i in range(1, len(self.x_train)):
            dist=euc(row, x_train[i])
            if(dist < best_distance):
                best_distance=dist
                best_index=i

        return self.y_train[best_index]

# load iris data
iris=datasets.load_iris()

x=iris.data     # data
y=iris.target   # labels

# spliting the data as training and testing data
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)

# writing classifier
my_classifier = ScrappyKNN()

# train the classifier
my_classifier.fit(x_train, y_train)

# make predictions based on the test data
predictions = my_classifier.predict(x_test)

# check the accuracy of the predictions
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))