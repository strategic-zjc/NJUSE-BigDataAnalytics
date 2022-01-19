from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
class decision_tree_classifier:
    def __init__(self):
        self.iris = load_iris()
        self.criterion = "gini"
        self.splitter = "best"
        self.random_state = None
    def __get_max_depth(self, x_train, y_train):
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)
        acc_prob = 0.
        max_depth = 0
        acc_list = []
        for i in range(2,11):
            clf = tree.DecisionTreeClassifier(criterion=self.criterion,
                                              random_state=self.random_state,
                                              splitter=self.splitter,
                                              max_depth= i)
            clf.fit(x_train, y_train)
            accuracy = clf.score(x_val, y_val)
            acc_list.append(accuracy)
            if(accuracy >= acc_prob):
                acc_prob = accuracy
                max_depth = i
        plt.plot(range(2, 11), acc_list, color="red", label="max_depth")
        plt.legend()
        plt.show()
        return max_depth
    def __get_min_samples_leaf(self, x_train, y_train):
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)
        acc_prob = 0.
        min_samples_leaf = 0
        acc_list = []
        for i in range(2,11):
            clf = tree.DecisionTreeClassifier(criterion=self.criterion,
                                              random_state=self.random_state,
                                              splitter=self.splitter,
                                              min_samples_leaf=i)
            clf.fit(x_train, y_train)
            accuracy = clf.score(x_val, y_val)
            acc_list.append(accuracy)
            if (accuracy >= acc_prob):
                acc_prob = accuracy
                min_samples_leaf = i
        plt.plot(range(2, 11), acc_list, color="red", label="min_samples_leaf")
        plt.legend()
        plt.show()
        return min_samples_leaf
    def __get_min_samples_split(self,x_train, y_train):
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)
        acc_prob = 0.
        min_samples_split = 0
        acc_list = []
        for i in range(2, 11):
            clf = tree.DecisionTreeClassifier(criterion=self.criterion,
                                              random_state=self.random_state,
                                              splitter=self.splitter,
                                              min_samples_split=i)
            clf.fit(x_train, y_train)
            accuracy = clf.score(x_val, y_val)
            acc_list.append(accuracy)
            if (accuracy >= acc_prob):
                acc_prob = accuracy
                min_samples_split = i
        plt.plot(range(2, 11), acc_list, color="red", label="min_samples_split")
        plt.legend()
        plt.show()
        return min_samples_split

    def fit(self, x_train, y_train):
        self.max_depth = self.__get_max_depth(x_train, y_train)
        self.min_samples_leaf = self.__get_min_samples_leaf(x_train,y_train)
        self.min_samples_split = self.__get_min_samples_split(x_train,y_train)
        self.clf = tree.DecisionTreeClassifier(criterion=self.criterion,
                                               random_state=self.random_state,
                                               splitter=self.splitter,
                                               max_depth=self.max_depth,
                                               min_samples_leaf=self.min_samples_leaf,
                                               min_samples_split=self.min_samples_split)
        self.clf.fit(x_train,y_train)
        plt.figure(figsize=(8, 8))
        tree.plot_tree(self.clf, filled='True',
                       feature_names=['花萼长', '花萼宽', '花瓣长', '花瓣宽'],
                       class_names=['山鸢尾', '变色鸢尾', '维吉尼亚鸢尾'])
        plt.savefig("./decision_tree.png", bbox_inches="tight", pad_inches=0.0)
        return
    def score(self,x_test, y_test):
        return self.clf.score(x_test,y_test)
    def predict(self,X):
        return self.clf.predict(X)

def get_data():
    iris = load_iris()
    x_train = iris.data
    y_train = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = get_data()
model = decision_tree_classifier()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("the accuracy is {}".format(score))

# 真正得到的模型
clf = model.clf

