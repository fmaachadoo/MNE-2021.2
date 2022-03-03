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
iteration_limit = 2


class ZeroOfAFunction:
    """ Implementation of function that finds the zero of a real function"""

    def __init__(self, function, interval: List[float], precision: List[float],
                 method: str, initial_guess=0.5):
        self.precision = precision
        if len(precision) != 0:
            self.single_precision = True
        else:
            self.single_precision = False
        self.interval = interval
        self.method = method
        #self.initial_guess = random.randint(interval[0], interval[1])
        self.initial_guess = initial_guess
        self.function = lambdify(x, function)
        self.derivative_function = lambdify(x, function.diff(x))
        self.string_derivative_function = function.diff(x)

    def f(self, x_):
        return self.function(x_)

    def f_(self, x_):
        return self.derivative_function(x_)

    def do_magick(self):
        if self.method == 'bisection':
            return self.bisection_method()
        elif self.method == 'newton_raphson':
            return self.newton_raphson_method(self.initial_guess)
        else:
            raise NotImplementedError

    def plot_graph(self):
        x_ = np.linspace(-10, 5, 100)
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

    def newton_raphson_method(self, x) -> float:
        global iteration
        x = round(x - self.f(x) / self.f_(x), ROUND_PRECISION)
        print(
            "k = " + str(iteration) + " | x" + str(iteration) + " = " + str(x) + " | f(" + str(
                x) + ") = " + str(self.f(x)))
        # if self.f(x) < E:
        #     return x
        # Remember to change iteration limit
        if iteration == iteration_limit:
            return x
        iteration += 1

        return self.newton_raphson_method(x)


if __name__ == '__main__':
    # function = (cos(x) * -1) + (E ** (x * -2))
    # function = x * ln(x) - 3.2
    # function = sqrt(x) - (E ** (x * -1))
    function = (x ** 4) - (2 * x ** 3) + (4 * x) - 1.6
    solver = ZeroOfAFunction(function=function, interval=[0, 1],
                             precision=[0.00000001], method='newton_raphson')
    # solver.plot_graph()

    print(f'f(x): {function}')
    print(f"f'(x): {solver.string_derivative_function}")
    print(solver.do_magick())
