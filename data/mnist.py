# from six.moves import cPickle as pickl
import pickle
import gzip
import numpy as np

def encode_label(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def shape_data(data):
    features = [np.reshape(x, (784, 1)) for x in data[0]]
    labels = [encode_label(y) for y in data[1]]

    return list(zip(features, labels))

def load_data():
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        train, validation, test = pickle.load(f, encoding="latin1")
    
    return shape_data(train), shape_data(validation), shape_data(test)

def average_digit(data, digit):
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)

    return np.average(filtered_array, axis=0)

def plot_eight():
    train, validate, test = load_data()
    avg_8 = average_digit(train, 8)

    img = np.reshape(avg_8, (28, 28))

    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.show()