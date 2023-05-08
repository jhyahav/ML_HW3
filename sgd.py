#################################
# Your name: Jonathan Yahav
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt
from scipy.special import expit

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
    return SGD(data, labels, C, eta_0, T, update_hinge_classifier)



def SGD_log(data, labels, C, eta_0, T, norm_mode=False):
    """
    Implements SGD for log loss.
    """
    return SGD(data, labels, None, eta_0, T, update_log_classifier)

#################################

# Place for additional code


HINGE = True
LOG = False
def run_experiment(flag):
    SGD_variant = SGD_hinge if flag == HINGE else SGD_log
    tr_data, tr_labels, v_data, v_labels, test_data, test_labels = helper()
    eta_range = np.logspace(-5, 5, 11)
    eta_0 = find_best_eta_0(eta_range, tr_data, tr_labels, v_data, v_labels, 1, 1000, SGD_variant)
    print("Best η_0:", eta_0)
    if flag == HINGE:
        C_range = np.logspace(-6, 2, 11)
        C = find_best_C(C_range, tr_data, tr_labels, v_data, v_labels, eta_0, 1000)
        print("Best C:", C)
    else:
        C = None
    display_classifier(tr_data, tr_labels, C, eta_0, 20000, test_data, test_labels, SGD_variant)
    if flag == LOG:
        plot_norm(tr_data, tr_labels, eta_0, 20000)

def find_best_eta_0(eta_range, tr_data, tr_labels, v_data, v_labels, C, T, SGD_variant):
    error_rates = [0] * eta_range.size
    for i in range(eta_range.size):
        for j in range(10):
            classifier = SGD_variant(tr_data, tr_labels, C, eta_range[i], T)
            error_rates[i] += cross_validate(classifier, v_data, v_labels) / 10
    accuracies = np.array([1 - error for error in error_rates])
    title = 'Average Accuracy on Validation Data as a Function of $η_{0}$'
    graph(title, eta_range, accuracies, '$η_{0}$', "Accuracy")
    return eta_range[np.argmin(error_rates)]

def find_best_C(C_range, tr_data, tr_labels, v_data, v_labels, eta_0, T):
    error_rates = [0] * C_range.size
    for i in range(C_range.size):
        for j in range(10):
            classifier = SGD_hinge(tr_data, tr_labels, C_range[i], eta_0, T)
            error_rates[i] += cross_validate(classifier, v_data, v_labels) / 10
    accuracies = np.array([1 - error for error in error_rates])
    title = 'Average Accuracy on Validation Data as a Function of $C$'
    graph(title, C_range, accuracies, '$C$', "Accuracy")
    return C_range[np.argmin(error_rates)]

def plot_norm(tr_data, tr_labels, eta_0, T):
    norm = SGD_log(tr_data, tr_labels, None, eta_0, T, norm_mode=True)
    title = "Norm of Classifier $w$ as a Function of SGD Iteration Number"
    graph(title, np.arange(0, T + 1), norm, "Iteration", "Norm of Classifier $w$")

def display_classifier(tr_data, tr_labels, C, eta_0, T, test_data, test_labels, SGD_variant):
    w = SGD_variant(tr_data, tr_labels, C, eta_0, T)
    error = 1 - cross_validate(w, test_data, test_labels)
    print("Error of best classifier:", error)
    plt.gray()
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.colorbar()
    plt.show()

def graph(title, xData, yData, xLabel, yLabel):
    plt.xscale("log")
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.plot(xData, yData, color="red")
    plt.title(title)
    plt.show()

def cross_validate(classifier, v_data, v_labels):
    error_rate = 0
    n = v_data.shape[0]
    for index in range(n):
        x_i, y_i = v_data[index], v_labels[index]
        error_rate += is_misprediction(classifier, x_i, y_i) / n
    return error_rate

def SGD(data, labels, C, eta_0, T, update_function, norm_mode=False):
    classifier = np.zeros(data.shape[1], dtype='float64')
    norm = 0 # np.zeros(T, dtype='float64') # TODO: implement updates
    for t in range(1, T+1):
        index = np.random.randint(1, data.shape[0])
        x_i, y_i = data[index], labels[index]
        eta_t = eta_0 / t
        classifier = update_function(classifier, x_i, y_i, eta_t, C)
    return classifier if not norm_mode else norm

def update_hinge_classifier(classifier, x_i, y_i, eta_t, C):
    print(classifier)
    print(x_i)
    classifier *= (1 - eta_t)  # this happens regardless of correctness of prediction
    classifier += is_hinge_miss(classifier, x_i, y_i) * eta_t * C * y_i * x_i
    return classifier

def update_log_classifier(classifier, x_i, y_i, eta_t, C):
    classifier += expit(np.dot(classifier, x_i) * (-y_i)) * x_i * y_i * eta_t
    return classifier

def is_hinge_miss(classifier, x_i, y_i):
    return y_i * np.dot(classifier, x_i) < 1

def is_misprediction(classifier, datapoint, label):
    predicted_label = sign(np.dot(classifier, datapoint))
    return label != predicted_label

def sign(x):
    return 1 if x >= 0 else -1


# run_experiment(HINGE)
run_experiment(LOG)

#################################
