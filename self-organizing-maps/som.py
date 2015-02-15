import numpy
import animals as anim

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
    units = 100
    input_size = len(anim.props.values()[0])

    weights = numpy.random.rand(units, input_size)
    for nbh in xrange(5, 1, -1): # Neighborhood sizes
        for x in anim.animals:
            winner, diff = find_winner(anim.props[x], weights)

            # Update weights of all nodes in the neighborhood
            for i in xrange(winner-nbh, winner+nbh):
                if not (0 <= i < units):
                    continue
                weights[i] += diff[i] * 0.2

    winner_per_animal = {a: find_winner(anim.props[a], weights)[0] for a in anim.animals}
    sorted_animals = sorted(anim.animals, key=lambda a : winner_per_animal[a])
    print map(lambda a : (a, winner_per_animal[a]), sorted_animals)