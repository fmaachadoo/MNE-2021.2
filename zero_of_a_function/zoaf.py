import random
from copy import deepcopy
from typing import List

from sympy import *
import matplotlib.pyplot as plt
import numpy as np
import math

ROUND_PRECISION = 5
x = Symbol('x')
iteration = 0
iteration_limit = 100000


class ZeroOfAFunction:
    """ Implementation of function that finds the zero of a real function"""

    def __init__(self, function, interval: List[float], precision: List[float],
                 method: str):
        self.precision = precision
        if len(precision) != 0:
            self.single_precision = True
        else:
            self.single_precision = False
        self.interval = interval
        self.method = method
        self.initial_guess = random.randint(interval[0], interval[1])
        self.function = lambdify(x, function)
        self.derivative_function = lambdify(x, function.diff(x))

    def f(self, x_):
        return self.function(x_)

    def f_(self, x_):
        return self.derivative_function(x_)

    def do_magick(self):
        if self.method == 'bisection':
            return self.bisection_method()
        else:
            raise NotImplementedError

    def plot_graph(self):
        x_ = np.linspace(0, 10, 100)
        plt.plot(x_, self.f(x_), label='value')
        plt.show()

    @staticmethod
    def is_positive(number):
        return True if number >= 0 else False

    def bisection_method(self) -> float:
        global iteration, iteration_limit
        while True:
            previous_interval = deepcopy(self.interval)
            x_ = round(np.mean(self.interval), ROUND_PRECISION)
            print(
                f'iteration = {str(iteration)} | x{str(iteration)} = {str(x_)} | f({str(x_)}) = '
                f'{str(self.f(x_))}'
            )
            # print(f'interval [a={self.interval[0]}, b={self.interval[1]}]')
            if abs(self.interval[1] - self.interval[0]) < self.precision[0]\
                    or iteration > iteration_limit:
                return x_
            if self.is_positive(self.f(self.interval[0])) != self.is_positive(
                    self.f(x_)):
                self.interval[1] = x_
                self.interval[0] += 5 * self.precision[0]

            elif x_ >= self.interval[0]:
                self.interval[0] = x_
            if previous_interval == self.interval:
                return x_

            iteration += 1


if __name__ == '__main__':
    #function = (cos(x) * -1) + (E ** (x * -2))
    function = x * ln(x) - 3.2
    solver = ZeroOfAFunction(function=function, interval=[2, 3],
                             precision=[0.0001], method='bisection')
    print(solver.do_magick())
