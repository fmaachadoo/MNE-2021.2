import random
from copy import deepcopy
from typing import List

from sympy import *
import csv
import matplotlib.pyplot as plt
import numpy as np
import math

ROUND_PRECISION = 5
x = Symbol('x')
iteration = 0
iteration_limit = 2


TOURISTS_ARRIVAL_BY_MONTH_CSV_PATH = 'tourists_arrival_by_month.csv' # https://www.gov.br/turismo/pt-br/acesso-a-informacao/acoes-e-programas/observatorio/anuario-estatistico/anuario-estatistico-de-turismo-2021-ano-base-2020/AnurioEstatsticodeTurismo2021AnoBase2020_2ED.pdf

class DataExtrapolation:
    """Implementation of function that extrapolate data of a given dataset"""

    def __init__(self, dataset_path):
        self.dataset = self.read_dataset(dataset_path)
        import pdb;pdb.set_trace()

    @staticmethod
    def read_dataset(dataset_path):
        dataset = {}
        with open(dataset_path, encoding='utf-16') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                if row[1].isnumeric():
                    dataset[row[0]] = row[1]
        return dataset

    @staticmethod
    def is_positive(number):
        return True if number >= 0 else False


if __name__ == '__main__':
    tourists = DataExtrapolation(dataset_path=TOURISTS_ARRIVAL_BY_MONTH_CSV_PATH)
    # tourists.plot_graph()
