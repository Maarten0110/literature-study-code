from random import uniform
import random
from quartic_action import create_random_quartic_coupling, QuarticProbabilityDistribution
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.use('TkAgg')
seed = 2
random.seed(seed)  #

if __name__ == "__main__":

    epsilon = np.linspace(0, 1, 8)
    K = np.array(
        [
            [1, 0],
            [0, 1]
        ]
    )
    V = create_random_quartic_coupling(2, lambda: uniform(-1, 1))
    print('seed: ' + str(seed) + '\n')
    print('K =\n' + str(K) + '\n')
    print('V =\n' + str(V))

    N = 100
    bounds = 2
    coords = np.linspace(-bounds, bounds, N + 1)
    fig = plt.figure(layout="constrained", figsize=(8, 12))
    plot_number = 0
    dists = []
    for eps in epsilon:
        dist = np.zeros((N + 1, N + 1))
        qpd = QuarticProbabilityDistribution(K, V, eps)

        for i, x in enumerate(coords):
            for j, y in enumerate(coords):
                z = np.array([x, y])
                dist[i, j] = qpd.compute(z)

        dists.append(dist)

    max_value = max([np.max(x) for x in dists])
    min_value = min([np.min(x) for x in dists])
    levels = np.linspace(min_value, max_value, 8)

    for i in range(len(epsilon)):
        dist = dists[i]
        ax = fig.add_subplot(4, 2, i+1, adjustable='box', aspect=1)
        cp = ax.contour(coords, coords, dist, levels=levels, vmax=max_value, vmin=min_value, colors='black')
        ax.clabel(cp, inline=True, fontsize=6)
        ax.set_title('ε = ' + str(round(epsilon[i], 2)))

    fig.suptitle("A quartic action at different values of ε.")
    plt.show()
