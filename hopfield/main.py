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
        assert utils.samePattern(pattern, updated_pattern), "Simple patterns are not fixpoints"

    # Test if the network will recall stored patterns from distorted versions
    for pattern in patterns:
        distorted_pattern = utils.flipper(pattern, len(pattern) / 2)
        print pattern, distorted_pattern
        for i in xrange(2):
            distorted_pattern = utils.update(weights, distorted_pattern)
        print pattern, distorted_pattern
        assert utils.samePattern(pattern, distorted_pattern)

    print "'Small patterns' experiment succesfull!"

def restoring_images():
    patterns = [figs.p1, figs.p2, figs.p3]
    weights = utils.learn(patterns)

    print "! Pattern recovery"
    for i, (original, noisy) in enumerate([(figs.p1, figs.p11), (figs.p2, figs.p22)]):
        recovered_pattern = utils.update(weights, noisy)

        if utils.samePattern(recovered_pattern, original):
            print "  . Correctly recovered pattern {}".format(i+1)
        else:
            print "  . Couldn't recover pattern {}".format(i+1)

    sequential_hopfield(weights, figs.p22, figs.p2)

def sequential_hopfield(weights, noisy, original):
    for i in xrange(40):
        for _ in xrange(100):
            utils.updateOne(weights, noisy)
        show_pattern(noisy)
        if utils.samePattern(noisy, original):
            print "Recovered the expected pattern."
            break

if __name__ == '__main__':
    #small_patterns()
    #restoring_images()
    pass