import matplotlib.pyplot as plt
import numpy as np

from mlp import mlp

def generate_points(number, mean, std):
    return mean + np.random.randn(number, 2) * std

def plot_points(points, style='.'):
    plt.plot(points[:,0], points[:,1], style)

def plot_boundary(perceptron):
    xrange = np.arange(-4, 4, 0.1)
    yrange = np.arange(-4, 4, 0.1)
    xgrid, ygrid = np.meshgrid(xrange, yrange)

    npoints = xgrid.shape[0] * xgrid.shape[1]
    xcoords = xgrid.reshape((npoints, 1))
    ycoords = ygrid.reshape((npoints, 1))
    samples = np.concatenate((xcoords, ycoords), axis=1)

    ones = np.ones(xcoords.shape)
    samples = np.concatenate((samples, ones), axis=1)

    indicator = perceptron.mlpfwd(samples)
    indicator = indicator.reshape(xgrid.shape)

    plt.contour(xrange, yrange, indicator, (0.5,))

if __name__ == '__main__':
    # Generating the points
    class_a = generate_points(10, mean=(10, 10), std=4)
    class_b = generate_points(10, mean=(-10, -10), std=4)

    # Plotting the points
    plt.figure()
    plot_points(class_a)
    plot_points(class_b)
    #plt.show()

    # Generating the concatenated matrix
    class_a = np.hstack([class_a, np.zeros((class_a.shape[0], 1))])
    class_b = np.hstack([class_b, np.ones((class_b.shape[0], 1))])
    data = np.vstack([class_a, class_b])

    # MLP training
    inputs, target = data[:, 2:], np.reshape(data[:, 2], (data.shape[0], 1))
    perceptron = mlp(inputs, target, nhidden=3)

    perceptron.mlptrain(inputs, target, eta=0.5, niterations=100)

    plot_boundary(perceptron)

    plt.show()