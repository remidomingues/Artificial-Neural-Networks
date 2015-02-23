import numpy as np
import matplotlib.pyplot as plt

import figs
import utils

x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1])
x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1])
x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1])

def show_pattern(pattern):
    plt.matshow(pattern.reshape((32, 32)))
    plt.show()

# 4) Experiments

def small_patterns():
    patterns = [x1, x2, x3]
    weights = utils.learn(patterns)

    # Testing that the patterns are "fixpoints"
    for pattern in patterns:
        updated_pattern = utils.update(weights, pattern)
        assert utils.samePattern(pattern, updated_pattern)

    # Test if the network will recall stored patterns from distorted versions
    for pattern in patterns:
        distorted_pattern = utils.flipper(pattern, len(pattern) / 2)
        for i in xrange(5):
            distorted_pattern = utils.update(weights, distorted_pattern)
        print pattern, distorted_pattern
        assert utils.samePattern(pattern, distorted_pattern)

    print "'Small patterns' experiment succesfull!"

if __name__ == '__main__':
    small_patterns()