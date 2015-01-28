import matplotlib.pyplot as plt
import numpy as np

from mlp import mlp

def generate_points(number, mean, std):
    res = np.random.randn(number, 2)
    res = np.array(map(lambda xy:( abs(xy[0]) + mean[0], abs(xy[1]) + mean[1]), res))
    return res

def plot_points(points, color, style='.'):
    plt.plot(points[:,0], points[:,1], style, color=color)

def plot_boundary(perceptron, inputs, color='#f1312d', linewidth=1):
    min_x, max_x = min(inputs[:, 0]), max(inputs[:, 0])
    min_y, max_y = min(inputs[:, 1]), max(inputs[:, 1])
    xrange = np.arange(min_x - 2, max_x + 2, 0.1)
    yrange = np.arange(min_y - 2, max_y + 2, 0.1)
    xgrid, ygrid = np.meshgrid(xrange, yrange)

    npoints = xgrid.shape[0] * xgrid.shape[1]
    xcoords = xgrid.reshape((npoints, 1))
    ycoords = ygrid.reshape((npoints, 1))
    samples = np.concatenate((xcoords, ycoords), axis=1)

    ones = -np.ones(xcoords.shape)
    samples = np.concatenate((samples, ones), axis=1)

    indicator = perceptron.mlpfwd(samples)
    indicator = indicator.reshape(xgrid.shape)

    plt.contour(xrange, yrange, indicator, (0.5,), colors=color, linewidths=linewidth)

if __name__ == '__main__':
    # Generating the points
    m_a, m_b = 0, 1
    std_a, std_b = 1., .5
    n_a, n_b = 100, 100

    class_a = (m_a, m_a) + np.random.randn(n_a, 2) * std_a
    class_b = m_b + np.random.randn(n_b, 2) * std_b

    # Generating the concatenated matrix
    class_a = np.hstack([class_a, np.zeros((class_a.shape[0], 1))])
    class_b = np.hstack([class_b, np.ones((class_b.shape[0], 1))])
    data = np.vstack([class_a, class_b])

    # MLP training
    inputs, target = data[:, :2], data[:, 2:]
    perceptron = mlp(inputs, target, nhidden=3)

    # Plotting
    plt.figure()
    plot_points(class_a, color='blue')
    plot_points(class_b, color='red')
    N_ITERATIONS = 10000
    STEP = 40
    for i in xrange(STEP):
        perceptron.mlptrain(inputs, target, eta=0.01, niterations=N_ITERATIONS/STEP)
        plot_boundary(perceptron, inputs)

    plot_boundary(perceptron, inputs, color='green', linewidth=4)
    plt.show()