from collections import Counter

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
    for i, pattern in enumerate(patterns):
        updated_pattern = utils.update(weights, pattern)
        if utils.samePattern(pattern, updated_pattern):
            print "* Pattern #{} is a fixpoint, as expected.".format(i+1)
    print

    # Test if the network will recall stored patterns from distorted versions
    NUM_TRIALS = 100
    for n in xrange(1, 4):
        print "# Recovering from {} flip(s):".format(n)
        for i, pattern in enumerate(patterns):
            success = 0
            for _ in xrange(NUM_TRIALS):
                distorted_pattern = utils.flipper(pattern, n)
                for j in xrange(500):
                    utils.updateOne(weights, distorted_pattern)

                if utils.samePattern(pattern, distorted_pattern):
                    success += 1
            print "  - Pattern #{}: {}/{} recoveries were succesful.".format(i+1, success, NUM_TRIALS)
        print

    # Finding unexpected attractors
    attractors = set()
    for i in xrange(1000):
        pattern = utils.rndPattern(len(patterns[0]))
        for _ in xrange(100):
            utils.updateOne(weights, pattern)
        if not any(np.all(pattern == p) for p in patterns):
            attractors.add(tuple(pattern.tolist()))

    print "# Unexpected attractors:"
    print '\n'.join(map(str, attractors))
    print

    print "'Small patterns' experiment succesfull!"

def restoring_images():
    patterns = [figs.p1, figs.p2, figs.p3]
    weights = utils.learn(patterns)

    print "! Pattern recovery ( 1 & 2 )"
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
    print
    #sequential_hopfield(weights, figs.p22, figs.p2, num_iter=3000, display=300)

    # Testing recovering distorted patterns
    pattern = figs.p1
    attractors = set()
    print "! Pattern recovery with varying distortion:"
    for n in xrange(1, len(pattern) - 1, 10):
        print "  * n = {}/{}".format(n, len(pattern))
        for trial in xrange(10):
            noisy = utils.flipper(pattern, n)
            for l in xrange(20000):
                utils.updateOne(weights, noisy)
                if l % 1000 == 0 and utils.samePattern(pattern, noisy):
                    break
            attractors.add(tuple(noisy.tolist()))
            if utils.samePattern(pattern, noisy):
                break

        if utils.samePattern(pattern, noisy):
            print "   . Correctly recovered the pattern (on at least one of the trials)"
        else:
            print "   * Couldn't recover the pattern, stopping."
            break


    # Energy at the different attractors
    x = Counter()
    for attr in attractors:
        energy = utils.energy(weights, np.array(attr))
        x[energy] += 1
    plt.plot(x.keys(), x.values(), 'b.')
    plt.title("Energy at different attractors")
    plt.show()

    # Studying the change of energy at each iteration
    noisy = utils.flipper(figs.p1, 40)
    iterations = 5000
    iterations = range(iterations)
    energies = []
    for iteration in iterations:
        energies.append(utils.energy(weights, noisy))
        utils.updateOne(weights, noisy)
    plt.plot(iterations, energies, '-b')
    plt.title("Evolution of the energy at each iteration")
    plt.show()


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
    sequential_hopfield(weights, pattern, num_iter=10000, display=500)

    # Make the random weight matrix symmetric
    weights = 0.5 * (weights + weights.transpose())

    # Randomly update the pattern
    pattern = np.copy(figs.p1)

    #show_pattern(pattern)
    sequential_hopfield(weights, pattern, num_iter=10000, display=500)

def capacity_benchmarks(patterns, force_recovery=False, updates=200, ntrials=10, bias=[0], plot=False):
    if force_recovery:
        print "! Capacity benchmarks: pattern_length={} updates={}, attempts={}".format(
           len(patterns[0]), len(patterns[0])*10, ntrials)
    else:
        print "! Capacity benchmarks: pattern_length={}".format(len(patterns[0]))

    for b in bias:
        if b != 0:
            print "=> BENCHMARKS: bias={}".format(b)
        # Increasing pattern memory
        for i in range(1, len(patterns)+1):
            recovery_failure = [0] * i
            if b == 0:
                weights = utils.learn(patterns[:i])
            else:
                weights = biasedLearn(patterns[:i])
            nmin = len(patterns[0])+1
            pmin = None

            # Applying benchmark on each pattern stored
            for p in xrange(i):
                pattern = patterns[p]
                recovered = False

                # Increasing pattern noise
                for n in range(1, len(patterns[0])+1):
                    if not force_recovery:
                        # Random noise
                        noisy_pattern = utils.flipper(pattern, n)
                        # Pattern recovery
                        noisy_pattern = utils.update(weights, noisy_pattern)

                        if not utils.samePattern(pattern, noisy_pattern):
                            recovery_failure[p] = n
                            break

                    else:
                        recovered = False

                        # Multiple attemps if failure
                        for t in xrange(ntrials):
                            # Random noise
                            noisy_pattern = utils.flipper(pattern, n)

                            # Pattern recovery
                            for j in xrange(len(patterns[0])*10):
                                if b == 0:
                                    utils.updateOne(weights, noisy_pattern)
                                else:
                                    biasedUpdateOne(weights, noisy_pattern, b)

                            if utils.samePattern(pattern, noisy_pattern):
                                recovered = True
                                break
                            else:
                                if n < nmin:
                                    nmin = n
                                    pmin = p+1
                                recovery_failure[p] = n

                        if not recovered:
                            break

            if force_recovery:
                print ("{} stored - All patterns recovered until {} (p{} failed) - Last failure at {} by p{}\n"+
                    "First attempt failed by p{} at {}\nDetails: {}").format(
                    i, min(recovery_failure), recovery_failure.index(min(recovery_failure)),
                    max(recovery_failure), recovery_failure.index(max(recovery_failure)),
                    nmin, pmin, recovery_failure)
            else:
                print "{} stored - All patterns recovered until {} (p{} failed) - Last failure at {} by p{}\nDetails: {}".format(
                    i, min(recovery_failure), recovery_failure.index(min(recovery_failure)),
                    max(recovery_failure), recovery_failure.index(max(recovery_failure)), recovery_failure)

def quantitative_plot(patterns, bias=None):
    FLIPPED = 30
    x = []
    y = []
    for n in xrange(1, len(patterns) + 1):
        x.append(n)
        considered_patterns = patterns[:n]
        if bias:
            weights = biasedLearn(considered_patterns)
        else:
            weights = utils.learn(considered_patterns)
        recovered = 0
        for p in considered_patterns:
            for trial in xrange(10):
                noisy = utils.flipper(p, FLIPPED)
                for _ in xrange(10 * len(p)):
                    if bias:
                        biasedUpdateOne(weights, noisy, bias)
                    else:
                        utils.updateOne(weights, noisy)
                    if utils.samePattern(noisy, p):
                        break

            if utils.samePattern(noisy, p):
                recovered +=1
        y.append(recovered)

    plt.plot(x, y)
    plt.title("Evolution of capacity with the number of patterns")
    plt.show()

def getRandomPatterns(n, length, bias=0):
    return np.array([biasedRandomPattern(length, bias) for _ in xrange(n)])

def biasedRandomPattern(n, bias=0):
    "Create a random pattern of length n"
    return np.sign(np.random.randn(n) + bias)

def biasedLearn(patterns):
    "Use Hebbs rule to find the weight matrix for a list of patterns"
    n = len(patterns[0])
    w = np.zeros((n, n))

    # Substract average activity to set mean at 0
    avg = sum(sum(p) for p in patterns) / float(n * len(patterns))
    updated_patterns = patterns - avg

    # Hebbs rule
    for p in updated_patterns:
        w += np.outer(p, p)

    # Remove self connections
    return w - np.diag(np.diag(w))

def biasedUpdateOne(w, x, bias=0):
    "Update one element in x"
    i = np.random.randint(len(x))
    x[i] = bias + bias * np.sign(np.dot(w[i,:], x) - bias)

def sparsePatterns(n, length, activity=0.1):
    nactivity = int(length*activity)
    patterns = np.array([np.array([1]*nactivity + [0]*(length-nactivity)) for _ in xrange(n)])
    map(np.random.shuffle, patterns)
    return patterns

if __name__ == '__main__':
    # small_patterns()
    # restoring_images()
    # random_connectivity()

    ## Capacity
    # patterns = np.array([figs.p1, figs.p2, figs.p3, figs.p4, figs.p5, figs.p6, figs.p7, figs.p8, figs.p9])
    # capacity_benchmarks(patterns)
    # quantitative_plot(patterns)

    ## Quantitative measurements
    # patterns = getRandomPatterns(20, 128)
    # capacity_benchmarks(patterns)
    # capacity_benchmarks(patterns, force_recovery=True, updates=1000, ntrials=10)
    # quantitative_plot(patterns)

    ## Sparse patterns
    # patterns = getRandomPatterns(20, 128, bias=-0.5)
    #capacity_benchmarks(patterns, force_recovery=True, updates=1000, ntrials=10)
    # quantitative_plot(patterns)

    ## Sparse patterns
    sparse_patterns = sparsePatterns(10, 100, 0.1)
    bias = np.array(range(5, 7)) / 10.
    # capacity_benchmarks(sparse_patterns, force_recovery=True, updates=1000, ntrials=10, bias=bias)
    quantitative_plot(sparse_patterns)
    quantitative_plot(sparse_patterns, bias=0.5)
    pass
