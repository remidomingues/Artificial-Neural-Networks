import matplotlib.pyplot as plt
import numpy as np

from mlp import mlp

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

def run(n_a=100, m_a=0, std_a=1., n_b=100, m_b=4, std_b=1., nsteps=40, niterations=1000, nhidden=3, eta=0.01, class_b=None):
    # Generating the points
    class_a = m_a + np.random.randn(n_a, 2) * std_a
    if class_b == None:
        class_b = m_b + np.random.randn(n_b, 2) * std_b

    # Generating the concatenated matrix
    class_a = np.hstack([class_a, np.zeros((class_a.shape[0], 1))])
    class_b = np.hstack([class_b, np.ones((class_b.shape[0], 1))])
    data = np.vstack([class_a, class_b])

    # MLP training
    inputs, target = data[:, :2], data[:, 2:]
    perceptron = mlp(inputs, target, nhidden=nhidden)

    # Plotting
    plt.figure()
    plot_points(class_a, color='blue')
    plot_points(class_b, color='red')

    for i in xrange(nsteps):
        perceptron.mlptrain(inputs, target, eta=eta, niterations=niterations/nsteps)
        plot_boundary(perceptron, inputs)

    plot_boundary(perceptron, inputs, color='green', linewidth=4)
    plt.title(('ANN Decision boundary\nN_A={}, mean_A={}, std_A={}, N_B={}, mean_B={}, std_B={}\n' +
        'n_steps={}, n_iterations={}, eta={}').format(
        n_a, m_a, std_a, n_b, m_b, std_b, nsteps, niterations, eta))
    plt.show()

def tests():
    print 'n_a=100, m_a=0, std_a=1., n_b=100, m_b=4, std_b=1., nsteps=40, niterations=1000, nhidden=3, eta=0.01'
    run(n_a=100, m_a=0, std_a=1., n_b=100, m_b=4, std_b=1., nsteps=40, niterations=1000, nhidden=3, eta=0.01)
    print 'm_b=2'
    run(n_a=100, m_a=0, std_a=1., n_b=100, m_b=2, std_b=1., nsteps=40, niterations=1000, nhidden=3, eta=0.01)
    print 'std_a=2, std_b=2'
    run(n_a=100, m_a=0, std_a=2., n_b=100, m_b=2, std_b=2., nsteps=40, niterations=1000, nhidden=3, eta=0.01)
    print 'n_a=1000, n_b=1000'
    run(n_a=1000, m_a=0, std_a=2., n_b=1000, m_b=3, std_b=2., nsteps=40, niterations=1000, nhidden=3, eta=0.01)
    print 'eta=0.001'
    run(n_a=100, m_a=0, std_a=2., n_b=100, m_b=3, std_b=2., nsteps=40, niterations=100, nhidden=3, eta=0.001)
    print 'eta=0.5'
    run(n_a=100, m_a=0, std_a=2., n_b=100, m_b=3, std_b=2., nsteps=40, niterations=100, nhidden=3, eta=0.1)

    n_b=200
    m_b1=(4, 4)
    m_b2=(0, 4)
    m_b3=(4, 0)
    std_b=1.
    class_b = m_b1 + np.random.randn(n_b, 2) * std_b
    class_b = np.vstack([class_b, m_b2 + np.random.randn(n_b, 2) * std_b])
    class_b = np.vstack([class_b, m_b3 + np.random.randn(n_b, 2) * std_b])

    print 'n_hidden=1'
    run(n_a=1000, m_a=0, std_a=1., class_b=class_b, nsteps=20, niterations=1000, nhidden=1, eta=0.01)
    print 'n_hidden=10'
    run(n_a=1000, m_a=0, std_a=1., class_b=class_b, nsteps=20, niterations=1000, nhidden=10, eta=0.01)
    print 'n_hidden=10, nsteps=100'
    run(n_a=1000, m_a=0, std_a=1., class_b=class_b, nsteps=100, niterations=1000, nhidden=10, eta=0.01)
    print 'n_hidden=10, niterations=5000'
    run(n_a=1000, m_a=0, std_a=1., class_b=class_b, nsteps=20, niterations=5000, nhidden=10, eta=0.01)


if __name__ == '__main__':
    tests()
