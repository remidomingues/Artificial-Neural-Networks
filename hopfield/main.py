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
    sequential_hopfield(weights, pattern, num_iter=300, display=300)

    # Make the random weight matrix symmetric
    weights = 0.5 * (weights + weights.transpose())

    # Randomly update the pattern
    pattern = np.copy(figs.p1)

    #show_pattern(pattern)
    sequential_hopfield(weights, pattern, num_iter=300, display=300)

def capacity(random=False, length=20, ntrials=10, npatterns=10):
    if not random:
        patterns = [figs.p1, figs.p2, figs.p3, figs.p4, figs.p5, figs.p6, figs.p7, figs.p8, figs.p9]
    else:
        patterns = [utils.rndPattern(length) for _ in xrange(npatterns)]

    # print "! Capacity benchmarks: pattern_length={} updates={}, attempts={}".format(
    #    len(patterns[0]), len(patterns[0])*10, ntrials)

    print "! Capacity benchmarks: pattern_length={}".format(len(patterns[0]))

    # Increasing pattern memory
    for i in range(1, len(patterns)+1):
        weights = utils.learn(patterns[:i])
        nmin = len(patterns[0])+1
        nmax = -1

        # Applying benchmark on each pattern stored
        for pattern in patterns:
            recovered = False

            # Increasing pattern noise
            for n in range(1, len(patterns[0])+1):
                # recovered = False

                # Multiple attemps if failure
                # for t in xrange(ntrials):
                noisy_pattern = utils.flipper(pattern, n)
                    # Pattern recovery
                    # for j in xrange(len(patterns[0])*10):
                    #     utils.updateOne(weights, noisy_pattern)
                noisy_pattern = utils.update(weights, noisy_pattern)

                if not utils.samePattern(pattern, noisy_pattern):
                    # recovered = True
                    if n < nmin:
                        nmin = n
                    break
                        # break
                    # elif n < nmin:
                    #     nmin = n
                    # elif n > nmax:
                    #     nmax = n

                # if not recovered:
                #     break

            if not recovered:
                break
        # print "{} stored - first failure at {}, failed recovery at {} ({} attempts)".format(i, nmin, nmax, ntrials)
        print "{} stored - failed recovery at {}".format(i, nmin)

if __name__ == '__main__':
    # small_patterns()
    # restoring_images()
    # random_connectivity()
    capacity()
    # capacity(random=True, length=64, npatterns=15)
    pass
