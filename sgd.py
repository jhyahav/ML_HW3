#################################
# Your name: Jonathan Yahav
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt
from scipy.special import softmax

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    classifier = np.zeros(data.shape[1])
    for t in range(1, T+1):
        index = np.random.randint(1, data.shape[0])
        x_i, y_i = data[index], labels[index]
        eta_t = eta_0 / t
        classifier = update_hinge_classifier(classifier, x_i, y_i, eta_t, C)
    return classifier



def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    # TODO: Implement me
    pass

#################################

# Place for additional code
def run_hinge_experiment():
    tr_data, tr_labels, v_data, v_labels, test_data, test_labels = helper()
    eta_range = np.logspace(-5, 5, 11)
    eta_0 = find_best_eta_0(eta_range, tr_data, tr_labels, v_data, v_labels, 1, 1000)
    print("Best η_0:", eta_0)
    C_range = np.logspace(-5, 5, 11)
    C = find_best_C(C_range, tr_data, tr_labels, v_data, v_labels, eta_0, 1000)
    print("Best C:", C)
    display_classifier(tr_data, tr_labels, C, eta_0, 20000)

def find_best_eta_0(eta_range, tr_data, tr_labels, v_data, v_labels, C, T):
    error_rates = [0] * 10
    for i in range(eta_range.size):
        for j in range(10):
            classifier = SGD_hinge(tr_data, tr_labels, C, eta_range[i], T)
            error_rates[i] += cross_validate(classifier, v_data, v_labels) / 10
    accuracies = np.array([1 - error for error in error_rates])
    title = '$Average Accuracy on Validation Data as a Function of η_{0}$'
    graph(title, eta_range, accuracies, '$η_{0}$', "Accuracy")
    return eta_range[np.argmin(error_rates)]

def find_best_C(C_range, tr_data, tr_labels, v_data, v_labels, eta_0, T):
    error_rates = [0] * 10
    for i in range(C_range.size):
        for j in range(10):
            classifier = SGD_hinge(tr_data, tr_labels, C_range[i], eta_0, T)
            error_rates[i] += cross_validate(classifier, v_data, v_labels) / 10
    accuracies = np.array([1 - error for error in error_rates])
    title = '$Average Accuracy on Validation Data as a Function of C$'
    graph(title, C_range, accuracies, "C", "Accuracy")
    return C_range[np.argmin(error_rates)]

def display_classifier(data, labels, C, eta_0, T):
    w = SGD_hinge(data, labels, C, eta_0, T)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')


def graph(title, xData, yData, xLabel, yLabels):
    plt.xlabel(xLabel)
    plt.plot(xData, yData, label=yLabels, color="red")
    plt.title(title)
    plt.show()

def cross_validate(classifier, v_data, v_labels):
    error_rate = 0
    n = v_data.shape[0]
    for index in range(n):
        x_i, y_i = v_data[index], v_labels[index]
        error_rate += is_misprediction(classifier, x_i, y_i) / n
    return error_rate

def update_hinge_classifier(classifier, x_i, y_i, eta_t, C):
    classifier *= (1 - eta_t)  # this happens regardless of correctness of prediction
    classifier += is_hinge_miss(classifier, x_i, y_i) * eta_t * C * y_i * x_i
    return classifier

def is_hinge_miss(classifier, x_i, y_i):
    return y_i * np.dot(classifier, x_i) < 1

def is_misprediction(classifier, datapoint, label):
    predicted_label = sign(np.dot(classifier, datapoint))
    return label != predicted_label

def sign(x):
    return 1 if x >= 0 else -1


run_hinge_experiment()

#################################
