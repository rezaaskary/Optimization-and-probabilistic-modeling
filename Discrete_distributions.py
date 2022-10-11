import numpy as np
from matplotlib.pyplot import plot, grid, figure, show


class DiscreteDistributions:
    def __init__(self,
                 n: int = None,
                 p: int = None) -> None:

        if isinstance(n, int):
            self.n = n
        elif n is None:
            self.n = None
        else:
            raise Exception('The lower bound is not specified correctly!')

        if isinstance(p, int):
            self.p = p
        elif p is None:
            self.p = None
        else:
            raise Exception('The lower bound is not specified correctly!')

    def visualize(self, lower_lim: float = -10, upper_lim: float = -10):
        """
        Visualizing the probability distribution
        :param lower_lim: the lower limit used in plotting the probability distribution
        :param upper_lim: the upper limit used in plotting the probability distribution
        :return: a line plot from matplotlib library
        """
        x_m = np.linspace(start=lower_lim, stop=upper_lim, num=1000)
        y_m = list()
        for i in range(len(x_m)):
            y_m.append(self.pdf(x_m[i]))
        plot(list(x_m.ravel()), y_m)
        grid(which='both')
        show()
