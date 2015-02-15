import house

import math
import numpy
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

def find_winner(input_vec, weights):
    # Difference between x and all prototype vectors
    diff = input_vec - weights
    # Calculate sum of squared differences
    # for all prototypes simultaneously
    dist = numpy.sum(diff*diff , axis=1)
    # Locate the winner (index to smallest distance)
    winner = numpy.argmin(dist)
    return winner, diff

def toGrid(idx):
    # Return (row, col)
    return (idx % grid_width, idx // grid_width)

def toArray(i, j):
    return i + j * grid_width

if __name__ == '__main__':
    iterations = 30
    input_size = len(house.votes.values()[0])
    grid_width = 10 # Grid size = grid_width^2

    weights = numpy.random.rand(grid_width*grid_width, input_size)
    for nbh in xrange(grid_width/2-1, 0, -1): # Neighborhood sizes
        for k in xrange(iterations):
            for voter in house.votes.values():
                winner, diff = find_winner(voter, weights)

                # Update weights of all nodes in the neighborhood
                for i, j in zip(xrange(winner-nbh, winner+nbh), xrange(winner-nbh, winner+nbh)):
                    if not (0 <= i < grid_width) or not (0 <= j < grid_width):
                        continue
                    idx = toArray(i, j)
                    weights[idx] += diff[idx] * 0.2


    # Socialistiska (left) = S, V, MP
    # Borgerliga (right) = M, C, FP, KD
    winner_per_voter = [pylab.zeros((grid_width,grid_width)),
                        pylab.zeros((grid_width,grid_width))]
    party_value = {'X': 0, 'R': 7, 'D': 10}

    for voter, votes in house.votes.iteritems():
        winner = find_winner(votes, weights)[0]
        if(house.house[voter][1] is 'R'):
            winner_per_voter[0][toGrid(winner)] = party_value[house.house[voter][1]]
        elif(house.house[voter][1] is 'D'):
            winner_per_voter[1][toGrid(winner)] = party_value[house.house[voter][1]]

    # Plot
    pylab.matshow(winner_per_voter[0])
    pylab.matshow(winner_per_voter[1])
    pylab.show()
