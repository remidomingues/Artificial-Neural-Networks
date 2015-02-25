import numpy as np
import matplotlib.pyplot as plt

import figs
import utils

x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1])
x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1])
x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1])

def show_pattern(pattern, title=None):
    plt.matshow(pattern.reshape((32, 32)))
    if title:
        plt.title(title)
    plt.show()

def seq_converge(weights, pattern):
    last_energy = utils.energy(weights, pattern)
    while True:
        utils.updateOne(weights, pattern)
        energy = utils.energy(weights, pattern)
        if energy == last_energy:
            break
        last_energy = energy

# 4) Experiments

def small_patterns():
    patterns = [x1, x2, x3]
    weights = utils.learn(patterns)

    # Testing that the patterns are "fixpoints"
    for pattern in patterns:
        updated_pattern = utils.update(weights, pattern)
        assert utils.samePattern(pattern, updated_pattern), "Simple patterns are not fixpoints"

    # Test if the network will recall stored patterns from distorted versions
    NUM_TRIALS = 100
    for n in xrange(1, 4):
        print "# Recovering from {} flip(s):".format(n)
        for i, pattern in enumerate(patterns):
            success = 0
            for _ in xrange(NUM_TRIALS):
                distorted_pattern = utils.flipper(pattern, n)
                for j in xrange(1000):
                    utils.updateOne(weights, distorted_pattern)
                seq_converge(weights, distorted_pattern)
                if utils.samePattern(pattern, distorted_pattern):
                    success += 1
            print "  - Pattern #{}: {}/{} recoveries were succesful.".format(i+1, success, NUM_TRIALS)
        print
    print "'Small patterns' experiment succesfull!"

def restoring_images():
    patterns = [figs.p1, figs.p2, figs.p3]
    weights = utils.learn(patterns)

    print "! Pattern recovery"
    for i, (original, noisy) in enumerate([(figs.p1, figs.p11), (figs.p2, figs.p22)]):
        noisy = np.array(noisy)
        show_pattern(noisy, title="Noisy pattern #{}".format(i+1))
        for _ in xrange(10000):
            utils.updateOne(weights, noisy)
        show_pattern(noisy, title="Recovered pattern #{}".format(i+1))
        if utils.samePattern(noisy, original):
            print "  . Correctly recovered pattern {}".format(i+1)
        else:
            print "  . Couldn't recover pattern {}".format(i+1)

    sequential_hopfield(weights, figs.p22, figs.p2, num_iter=3000, display=300)

def sequential_hopfield(weights, noisy, original=None, num_iter=100, display=100):
    for i in xrange(1, 1 + num_iter):
        utils.updateOne(weights, noisy)
        # Display
        if i % display == 0:
            show_pattern(noisy, title="Pattern recovered after {} iterations.".format(i))
        if utils.samePattern(noisy, original):
            print "Recovered the expected pattern."
            break

def random_connectivity():
    # Generate a random weight matrix
    weights = np.random.randn(len(figs.p1), len(figs.p1))

    # Randomly update the pattern
    pattern = np.copy(figs.p1)
    #show_pattern(pattern)
    sequential_hopfield(weights, pattern, num_iter=100, display=100)

    # Make the random weight matrix symmetric
    weights = 0.5 * (weights + weights.transpose())

    # Randomly update the pattern
    pattern = np.copy(figs.p1)
    #show_pattern(pattern)
    sequential_hopfield(weights, pattern, num_iter=100, display=100)


if __name__ == '__main__':
    small_patterns()
    # restoring_images()
    # random_connectivity()
    pass