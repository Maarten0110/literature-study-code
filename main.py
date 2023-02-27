from random import uniform
import random
from quartic_action import create_random_quartic_coupling, QuarticProbabilityDistribution
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.use('TkAgg')
random.seed(8)  # V5: 13, 102, 3, 1101995 | V-1,1: 1, 8+bounds 2.5, 9+bounds 2.5

if __name__ == "__main__":

    epsilon = np.linspace(0, 1, 8)
    K = np.array(
        [
            [1, 0],
            [0, 1]
        ]
    )
    V = create_random_quartic_coupling(2, lambda: uniform(-1, 1))

    N = 100
    bounds = 2.5
    coords = np.linspace(-bounds, bounds, N + 1)
    fig = plt.figure(layout="constrained", figsize=(8, 12))
    plot_number = 0
    for plot_i in range(2):
        for plot_j in range(4):
            plot_number += 1
            print(plot_number)
            eps = epsilon[plot_number - 1]
            dist = np.zeros((N + 1, N + 1))
            qpd = QuarticProbabilityDistribution(K, V, eps)

            for i, x in enumerate(coords):
                for j, y in enumerate(coords):
                    z = np.array([x, y])
                    dist[i, j] = qpd.compute(z)

            ax = fig.add_subplot(4, 2, plot_number, adjustable='box', aspect=1)
            ax.contour(coords, coords, dist)
            ax.set_title('ε = ' + str(round(eps, 2)))

    plt.show()
