import matplotlib.pyplot as plt
import numpy as np

def generate_points(number, mean, std):
    return mean + np.random.randn(number, 2) * std

def plot_points(points, style='.'):
    plt.plot(points[:,0], points[:,1], style)

if __name__ == '__main__':
    # Generating the points
    class_a = generate_points(10, mean=(10, 10), std=4)
    class_b = generate_points(10, mean=(-10, -10), std=4)

    # Plotting the points
    plt.figure()
    plot_points(class_a)
    plot_points(class_b)
    plt.show()

    # Generating the concatenated matrix
    class_a = np.hstack([class_a, np.zeros((class_a.shape[0], 1))])
    class_b = np.hstack([class_b, np.ones((class_b.shape[0], 1))])
    data = np.vstack([class_a, class_b])
    print data