from matplotlib import pyplot as plt


def plot_histogram(tensor, color="orange", density=True, xlim=(0, 2500)):
    plt.hist(tensor.tolist(), bins=1000, color=color, density=density)
    plt.xlim(xlim)
    plt.show()
