import numpy
import animals as anim

units = 8
input_size = len(anim.props.values()[0])

w = numpy.random.rand(units, input_size)
for nbh in xrange(4, 1, -1): # Neighborhood sizes
    for x in anim.animals:
        # Difference between x and all prototype vectors
        diff = anim.props[x] - w
        # Calculate sum of squared differences
        # for all prototypes simultaneously
        dist = numpy.sum(diff*diff , axis=1)
        # Locate the winner (index to smallest distance)
        winner = numpy.argmin(dist)

        # Update weights of all nodes in the neighborhood
        for i in xrange(winner-nbh, winner+nbh):
            if not (0 <= i < units):
                continue
            w[i] += diff[i] * 0.2