import house

import numpy as np
import matplotlib.pyplot as plt

def find_winner(input_vec, weights):
    # Difference between x and all prototype vectors
    diff = input_vec - weights
    # Calculate sum of squared differences
    # for all prototypes simultaneously
    dist = np.sum(diff*diff , axis=1)
    # Locate the winner (index to smallest distance)
    winner = np.argmin(dist)
    return winner, diff

def toGrid(idx):
    # Return (row, col)
    return (idx % grid_width, idx // grid_width)

def toArray(i, j):
    return i + j * grid_width

if __name__ == '__main__':
    iterations = 2
    input_size = len(house.votes.values()[0])
    grid_width = 25 # Grid size = grid_width^2

    weights = np.random.rand(grid_width*grid_width, input_size)
    for nbh in xrange(grid_width/2-1, 0, -1): # Neighborhood sizes
        for k in xrange(iterations):
            for votes in house.votes.values():
                winner, diff = find_winner(votes, weights)

                # Update weights of all nodes in the neighborhood
                w_i, w_j = toGrid(winner)
                for i, j in zip(xrange(w_i-nbh, w_i+nbh), xrange(w_j-nbh, w_j+nbh)):
                    if not (0 <= i < grid_width) or not (0 <= j < grid_width):
                        continue
                    idx = toArray(i, j)
                    weights[idx] += diff[idx] * 0.2


    # Socialistiska (left) = S, V, MP
    # Borgerliga (right) = M, C, FP, KD
    winner_per_party = [np.zeros((grid_width, grid_width)),
                        np.zeros((grid_width, grid_width))]

    party_value = {'X': 0, 'R': 7, 'D': 10}
    winner_per_voter = dict()
    winners_list = {'X': [], 'R': [], 'D': []}
    for voter, votes in house.votes.iteritems():
        winner = find_winner(votes, weights)[0]
        party = house.house[voter][1]

        # Populating winner_per_voter and winners_list
        winner_per_voter[voter] = toGrid(winner)
        winners_list[party].append(toGrid(winner))

        # Populating the matrix
        if party == 'R':
            winner_per_party[0][toGrid(winner)] = party_value[party]
        elif party == 'D':
            winner_per_party[1][toGrid(winner)] = party_value[party]

    # Plot
    winners_list = {k : np.array(v) for k, v in winners_list.iteritems()}

    size = 16
    for (party, color) in [('X', 'y'), ('R', 'r'), ('D', 'b')]:
        plt.plot(winners_list[party][:, 0], winners_list[party][:, 1], '{}o'.format(color), markersize=size)
        size /= 2

    plt.matshow(winner_per_party[0])
    plt.matshow(winner_per_party[1])
    plt.show()