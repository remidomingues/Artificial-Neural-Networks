import cities

import numpy
import matplotlib.pyplot as plt

def find_winner(input_vec, weights):
    # Difference between x and all prototype vectors
    diff = input_vec - weights
    # Calculate sum of squared differences
    # for all prototypes simultaneously
    dist = numpy.sum(diff*diff , axis=1)
    # Locate the winner (index to smallest distance)
    winner = numpy.argmin(dist)
    return winner, diff


if __name__ == '__main__':
    units = 30
    iterations = 30
    input_size = len(cities.city[0])

    weights = numpy.random.rand(units, input_size)
    for nbh in xrange(units/2, 1, -1): # Neighborhood sizes
        for k in xrange(iterations):
            for city in cities.city:
                winner, diff = find_winner(city, weights)

                # Update weights of all nodes in the neighborhood
                for i in xrange(winner-nbh, winner+nbh):
                    actual_idx = i % len(weights)
                    weights[actual_idx] += diff[actual_idx] * 0.2

    winner_per_city = [find_winner(c, weights)[0] for c in cities.city]

    # Plot
    weights_plot = numpy.vstack((weights, numpy.array([weights[0]]))) # Looping plot
    plt.figure()
    plt.plot(cities.city[:, 0], cities.city[:, 1], 'or')
    plt.plot(weights_plot[:, 0], weights_plot[:, 1], 'b')
    plt.plot(weights_plot[:, 0], weights_plot[:, 1], '.b')
    plt.show()