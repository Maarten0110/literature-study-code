import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats

mpl.use('TkAgg')

if __name__ == "__main__":
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt

    # Define the x values for the plots
    x = np.linspace(-6, 6, 1000)

    # Define the three distributions
    distributions = [
        stats.norm(loc=0, scale=2.5).pdf(x),
        stats.norm(loc=0, scale=0.05).pdf(x)/8,
        stats.t(df=3, loc=0, scale=0.3).pdf(x)/2+0.05/(abs(x)*0.1+1),
    ]

    # Set the figure size and shape of the subplots
    fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=True, dpi=150)
    fig.subplots_adjust(wspace=0.05)

    # Loop through each subplot and plot the distribution
    for i in range(len(ax)):
        ax[i].plot(x, distributions[i], color='#4a86e8', linewidth=3)
        ax[i].set_ylim([0, 1])
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)

    # Show the plot
    plt.tight_layout()
    plt.show()
