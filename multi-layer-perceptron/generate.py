import matplotlib.pyplot as plt
import numpy as np

from mlp import mlp

def generate_points(number, mean, std):
    return mean + np.random.randn(number, 2) * std

def plot_points(points, style='.'):
    plt.plot(points[:,0], points[:,1], style)

def plot_boundary(perceptron, inputs):
    min_x, max_x = min(inputs[:, 0]), max(inputs[:, 0])
    min_y, max_y = min(inputs[:, 1]), max(inputs[:, 1])
    xrange = np.arange(min_x - 2, max_x + 2, 0.1)
    yrange = np.arange(min_y - 2, max_y + 2, 0.1)
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
    m_a, m_b = 0, 8
    std_a, std_b = 1., 1.
    class_a = generate_points(10, mean=(m_a, m_a), std=std_a)
    class_b = generate_points(10, mean=(m_b, m_b), std=std_b)

    # Generating the concatenated matrix
    class_a = np.hstack([class_a, np.zeros((class_a.shape[0], 1))])
    class_b = np.hstack([class_b, np.ones((class_b.shape[0], 1))])
    data = np.vstack([class_a, class_b])

    # MLP training
    inputs, target = data[:, :2], data[:, 2:]
    perceptron = mlp(inputs, target, nhidden=100)

    perceptron.mlptrain(inputs, target, eta=0.5, niterations=10000)


    # Plotting
    plt.figure()
    plot_points(class_a)
    plot_points(class_b)
    plot_boundary(perceptron, inputs)

    plt.show()