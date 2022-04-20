import random
from copy import deepcopy
from typing import List

from sympy import *
import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from datetime import datetime
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

    @staticmethod
    def read_dataset(dataset_path):
        dataset = {}
        with open(dataset_path, encoding='utf-16') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                if row[1].isnumeric():
                    dataset[row[0]] = row[1]
        return dataset

    def plot_dataset_graph(self):
        x = [datetime.strptime(d, '%Y-%m').date() for d in self.dataset.keys()]
        y = [int(v) for v in self.dataset.values()]
        plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(dates.DayLocator(interval=32))
        plt.plot(x, y, label='value')
        plt.gcf().autofmt_xdate()
        plt.xlabel(u"Chegada de turistas no brasil por mês")
        plt.ylabel(u"Data no formato Ano-mês")
        plt.show()

    @staticmethod
    def is_positive(number):
        return True if number >= 0 else False


if __name__ == '__main__':
    tourists = DataExtrapolation(dataset_path=TOURISTS_ARRIVAL_BY_MONTH_CSV_PATH)
    tourists.plot_dataset_graph()
    # tourists.plot_graph()
