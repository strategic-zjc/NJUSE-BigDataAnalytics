from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
class bpnn_classifier:
    def __init__(self):
        self.bpnn = None
        self.hidden_layer_size = 0
    def __get_hidden_layer_size(self, x_train, y_train):
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)
        acc_prob = 0.
        hidden_layer_size = 0
        acc_list = []
        for i in range(100, 1100, 100):
            clf = MLPClassifier(hidden_layer_sizes=(i,), solver='lbfgs', max_iter=2000)
            clf.fit(x_train, y_train)
            accuracy = clf.score(x_val, y_val)
            acc_list.append(accuracy)
            if (accuracy >= acc_prob):
                acc_prob = accuracy
                hidden_layer_size = i
        plt.plot(range(100, 1100, 100), acc_list, color="red", label="hidden layer size")
        plt.legend()
        plt.show()
        return hidden_layer_size
    def fit(self,x_train, y_train):
        self.hidden_layer_size = self.__get_hidden_layer_size(x_train, y_train)
        self.bpnn = MLPClassifier(hidden_layer_sizes=(self.hidden_layer_size,), solver='lbfgs',max_iter=2000)
        self.bpnn.fit(x_train,y_train)
    def score(self, x_test, y_test):
        return self.bpnn.score(x_test, y_test)


def get_data():
    iris = load_iris()
    x_train = iris.data
    y_train = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = get_data()


model = bpnn_classifier()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("the accuracy is {}".format(score))