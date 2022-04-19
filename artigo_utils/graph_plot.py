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

# Tourism Dataset follows the following format: key:`%Y-%m`(string date): `%f`(float number value)
# Chegadas de turistas internacionais ao Brasil por mês
TOURISM_DATASET = {
    '2019-01': 1, # Janeiro de 2019
    '2019-02': 1, # Fevereiro de 2019
    '2019-03': 1, # Março de 2019
    '2019-04': 1, # Abril de 2019
    '2020-05': 1, # Maio de 2019
    '2019-06': 1, # Junho de 2019
    '2019-07': 1, # Julho de 2019
    '2019-08': 1, # Agosto de 2019
    '2019-09': 1, # Setembro de 2019
    '2019-10': 1, # Outubro de 2019
    '2019-11': 1, # Novembro de 2019
    '2019-12': 1, # Dezembro de 2019
    '2020-01': 1, # Janeiro de 2020
    '2020-02': 1, # Fevereiro de 2020
    '2020-03': 1, # Março de 2020
    '2020-04': 1, # Abril de 2020
    '2020-05': 1, # Maio de 2020
    '2020-06': 1, # Junho de 2020
    '2020-07': 1, # Julho de 2020
    '2020-08': 1, # Agosto de 2020
    '2020-09': 1, # Setembro de 2020
    '2020-10': 1, # Outubro de 2020
    '2020-11': 1, # Novembro de 2020
    '2020-12': 1, # Dezembro de 2020
}


class DataExtrapolation:
    """Implementation of function that extrapolate data of a given dataset"""

    def __init__(self, dataset):
        self.dataset = dataset

    @staticmethod
    def is_positive(number):
        return True if number >= 0 else False

    

if __name__ == '__main__':
    function = (x ** 4) - (2 * x ** 3) + (4 * x) - 1.6
    solver = DataExtrapolation(dataset=dataset)
    # solver.plot_graph()
