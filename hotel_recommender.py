from __future__ import print_function

'''
@author Arjun Dhuliya and Siddharth Subramanian
'''

import pickle

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.metrics as metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle


def read_input_split_data():
    """
    reads data from 'train.csv' and splits it evenly into train and test data maintaining a ratio of 70-30
    writes the train and test data into csv
    :return: None
    """
    print('reading data...')
    # train = pd.read_csv('train.csv')
    # all_data = pd.read_csv('train.csv')
    all_data = pd.read_csv('out.csv')
    all_data["date_time"] = pd.to_datetime(all_data["date_time"])
    all_data["year"] = all_data["date_time"].dt.year
    all_data["month"] = all_data["date_time"].dt.month
    all_data["date_time"] = all_data["date_time"].dt.day
    all_data["srch_ci"] = pd.to_datetime(all_data["srch_ci"])
    all_data["srch_ci_year"] = all_data["srch_ci"].dt.year
    all_data["srch_ci_month"] = all_data["srch_ci"].dt.month
    all_data["srch_ci"] = all_data["srch_ci"].dt.day

    all_data["srch_co"] = pd.to_datetime(all_data["srch_co"])
    all_data["srch_co_year"] = all_data["srch_co"].dt.year
    all_data["srch_co_month"] = all_data["srch_co"].dt.month
    all_data["srch_co"] = all_data["srch_co"].dt.day
    print('Reading Done')
    for key in all_data.keys():
        all_data = all_data[pd.notnull(all_data[key])]

    # print(all.keys())
    hotel_id_set = set(all_data['hotel_cluster'])
    train = None
    test = None
    # all_data = all_data[all_data['is_booking'] == 1]
    # total = 0
    for hotel_id in hotel_id_set:
        flt = all_data[all_data['hotel_cluster'] == hotel_id]
        flt = shuffle(flt)
        l = len(flt)
        train_rows = int(l * 0.7)
        if train is None:
            train = flt[:train_rows]
            test = flt[train_rows:]
        else:
            train = pd.concat([train, flt[:train_rows]])
            test = pd.concat([test, flt[train_rows:]])
    print(train.shape)
    print(test.shape)
    print(all_data.shape)
    train.to_csv('train_naive.csv', index=False)
    test.to_csv('test_naive.csv', index=False)
    print("csv files written to train_naive.csv, test_naive.csv'")


def read_train_test_csv():
    """
    reads the train_naive and test_naive csv and returns as a tuple
    :return: train , test tuple
    """
    return pd.read_csv("train_naive.csv"), pd.read_csv("test_naive.csv")


def classify_random_forest(train, test):
    """
    classify using random forest
    :param train: train data
    :param test: test data
    :return: test_accuracy, train_accuracy
    """
    print("classify_random_forest", end="")
    print(" Model Fitting....")
    classifier = RandomForestClassifier(n_estimators=200)
    classifier.fit(train.ix[:, 0:30], train.ix[:, 30:].values.ravel())
    pickle.dump(classifier, open("RandomForest.dat", "wb"))
    res = classifier.predict(test.ix[:, 0:30])
    accuracy = metrics.accuracy_score(test.ix[:, 30:], res, normalize=True)
    print("Tested on testing data using model and accuracy: ", accuracy)
    res = classifier.predict(train.ix[:, 0:30])
    train_accuracy = metrics.accuracy_score(train.ix[:, 30:], res, normalize=True)
    print("Tested on Training data using model and accuracy: ", train_accuracy)
    return accuracy, train_accuracy


def classify_gaussian_nb(train, test):
    """
    classify using naive bayes
    :param train: train data
    :param test: test data
    :return: test_accuracy, train_accuracy
    """
    print("classify_gaussian_nb", end="")
    print(" Model Fitting....")
    classifier = GaussianNB()
    classifier.fit(train.ix[:, 0:30], train.ix[:, 30:].values.ravel())
    pickle.dump(classifier, open("GaussianNB.dat", "wb"))
    res = classifier.predict(test.ix[:, 0:30])
    accuracy = metrics.accuracy_score(test.ix[:, 30:], res, normalize=True)
    print("Tested on testing data using model and accuracy: ", accuracy)
    res = classifier.predict(train.ix[:, 0:30])
    train_accuracy = metrics.accuracy_score(train.ix[:, 30:], res, normalize=True)
    print("Tested on Training data using model and accuracy: ", train_accuracy)
    return accuracy, train_accuracy


def classify_gradient_boosting_classifier(train, test):
    """
    classify using gradient boosting
    :param train: train data
    :param test: test data
    :return: test_accuracy, train_accuracy
    """
    print("classify_gradient_boosting_classifier", end="")
    print(" Model Fitting....")
    classifier = GradientBoostingClassifier()
    classifier.fit(train.ix[:, 0:30], train.ix[:, 30:].values.ravel())
    pickle.dump(classifier, open("gradient.dat", "wb"))
    res = classifier.predict(test.ix[:, 0:30])
    accuracy = metrics.accuracy_score(test.ix[:, 30:], res, normalize=True)
    print("Tested on testing data using model and accuracy: ", accuracy)
    res = classifier.predict(train.ix[:, 0:30])
    train_accuracy = metrics.accuracy_score(train.ix[:, 30:], res, normalize=True)
    print("Tested on Training data using model and accuracy: ", train_accuracy)
    return accuracy, train_accuracy


def classify_logistic_regression(train, test):
    """
    classify using logistic regression
    :param train: train data
    :param test: test data
    :return: test_accuracy, train_accuracy
    """
    print("classify_logistic_regression", end="")
    print(" Model Fitting....")
    classifier = LogisticRegression(multi_class='ovr', C=100)
    classifier.fit(train.ix[:, 0:30], train.ix[:, 30:].values.ravel())
    pickle.dump(classifier, open("Logistic.dat", "wb"))
    res = classifier.predict(test.ix[:, 0:30])
    accuracy = metrics.accuracy_score(test.ix[:, 30:], res, normalize=True)
    print("Tested on testing data using model and accuracy: ", accuracy)
    res = classifier.predict(train.ix[:, 0:30])
    train_accuracy = metrics.accuracy_score(train.ix[:, 30:], res, normalize=True)
    print("Tested on Training data using model and accuracy: ", train_accuracy)
    return accuracy, train_accuracy


def main():
    """
    main function performing the reading, fitting and classifying
    :return: None
    """
    # read_input_split_data()
    train, test = read_train_test_csv()
    print("train shape: \n", train.shape)
    print("test shape: \n", test.shape)
    # print("train keys: \n", train.keys())
    # print(train.head(1))
    # print("shape of 30 columns train:", train.ix[:, 30:].shape)
    performances = []
    performances.append(["Naive Bayes", classify_gaussian_nb(train, test)])
    performances.append(["Random Forest", classify_random_forest(train, test)])
    performances.append(["Logistic Regression", classify_logistic_regression(train, test)])
    performances.append(["Gradient Boosting", classify_gradient_boosting_classifier(train, test)])
    for val in performances:
        print(val)

if __name__ == '__main__':
    main()
