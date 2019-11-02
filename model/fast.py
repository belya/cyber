# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np

from scipy.sparse import random

from scipy import stats

import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize

import scipy.sparse as sps

from tqdm import tqdm_notebook

normal_rvs = stats.gaussian(25, loc=10).rvs

# # Uniformly distributed

test_matrix = random(100, 100, density=1)

plt.hist(test_matrix.todense().reshape(-1).tolist())

# # Normalized by rows

test_matrix = random(3, 4, density=1)

# Rows - senders
#
# Columns - receivers

test_matrix.todense()

test_matrix = normalize(test_matrix, norm='l1', axis=1)

test_matrix.todense()

# # Multiplied by agent balances

test_matrix.todense()

balances = (np.random.rand(test_matrix.shape[0]) > 0.5).astype(float)

balances

test_matrix.T.multiply(balances).T.todense()

# # Total outcomes and incomes

outcome = test_matrix.sum(axis=1).A1

outcome.shape

income = test_matrix.sum(axis=0).A1

income.shape

# # Matrix element-wise multiplication for comission

test_matrix.multiply(test_matrix)


def create_similar_random_matrix(coo_matrix):
    rows = coo_matrix.tocoo().row
    cols = coo_matrix.tocoo().col
    data = np.random.rand(len(rows))
    return sps.coo_matrix((data, (rows, cols)), shape=coo_matrix.shape)


create_similar_random_matrix(test_matrix).todense()


# # Random transactions (+ comission)

def do_random_transactions(balances, density=1, max_transaction_rate=0.01, max_comission_rate=0.1):
    agents_size = balances.shape[0]
    
    random_max_transaction_rates = max_transaction_rate * np.random.rand(balances.shape[0])
    random_total_transaction_rates = random(agents_size, agents_size, density=density)
    normalized_random_total_transaction_rates =  normalize(random_total_transaction_rates, norm='l1', axis=1)
    total_transaction_amounts = normalized_random_total_transaction_rates.T.multiply(random_max_transaction_rates * balances).T
    
    transactions_sended = (total_transaction_amounts > 0)
    random_comission_rates = max_comission_rate * create_similar_random_matrix(transactions_sended)
    comission_amounts = total_transaction_amounts.multiply(random_comission_rates)
    transaction_amounts = total_transaction_amounts - comission_amounts
    
    income = transaction_amounts.sum(axis=0).A1
    outcome = transaction_amounts.sum(axis=1).A1 + comission_amounts.sum(axis=1).A1
    return balances + income - outcome, comission_amounts.sum()


SIZE = 1000000

balances = np.random.rand(SIZE)

density = 1 / ((SIZE * SIZE) / (1 * 1024 * 1024))

density

balances, _ = calculate_new_balances(balances, density=density)

balances
