import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pymc3 as pm
import theano 
import theano.tensor as tt
import arviz as az
from pprint import pprint
from scipy import stats
import sqlite3 as sql
from os import path


def LoT_indices_to_operators(indices=None, use_whole_effective_LoT_indices=False, return_nested_list=False):
    """
    Parameters
    ----------
    indices: list of ints
        Indices of LoTs
    use_whole_effective_LoT_indices: bool
        Whether the passed indices correspond to the actual languages
        or if they are effective indices 
        (e.g. 0 isnt an LoT in normal encoding, but it is in effective indices encoding)
    return_nested_list: bool
        Whether to return a boolean df with the operators as columns
        or a nested list with the operator names for each LoT
    """
    if indices is None:
        if use_whole_effective_LoT_indices:
            L, category_i, cost_i = get_data()
            _, indices = get_extended_L_and_effective(L)
        else:
            indices = np.arange(2**9)
    if use_whole_effective_LoT_indices: 
        assert len(indices) == 838, 'Indices does not have the right length!'
        indices = indices[:len(indices)//2]
    else:
        # there can be at most n**9 indices 
        assert len(indices) <= 2**9, (
            'Too many indices. Are you using the whole effective_LoT_indices '
            'produced by get_extended_L_and_effective(L) below? '
            'That does not work because the second half of effective_LoT_indices refers '
            'to the alternative interpretation of 0/1 as true/false. '
            'Only use first half!'
        )
    lists = np.array([list(f'{x:09b}') for x in indices])
    columns = ['O','A','N','C','B','X','NA','NOR','NC']
    if use_whole_effective_LoT_indices:
        # repeat the indices twice to cover the second repetition of the
        # LoTs with the opposite interpretation of 0/1
        lists = np.tile(lists,(2,1))
        
    df = pd.DataFrame(lists,columns=columns,dtype=bool)
    
    if return_nested_list:
        # Go from df to a list of lists, where each sublist contains
        # the names of the operators in that LoT
        return [
            [a for a in sublist if a!=0]
            for sublist in
            np.where(df.values, df.columns.values[None], 0).tolist()
        ]
    else:
        return df


def get_data(path_L='../data/lengths_data.npy', path_learningdata='../data/learning_costs.pkl'):
    """
    Get the data, i.e. L and the learning data
    
    Return
    ------
    tuple of arrays
        L: 
        category_i: the index of the category for each cost in cost_i
        cost_i: learning costs (multiple ones for each category)
    """

    # L has shape (LoT, cat)
    # where the LoT index encodes the LoT
    # in the way described by the 
    # encoding file
    L = np.load(path_L)

    # note that learning costs of ANNs
    # are calculated for only half of the categories
    # because the other half is identical from the POW
    # of the networks
    learning_data = pd.read_pickle(path_learningdata)
    category_i, _, cost_i = learning_data.values.T

    return L, category_i, cost_i


def get_params_from_fpath(fpath):
    fname = path.splitext(path.basename(fpath))[0] 
    fname = fname.lstrip('report_')
    params = dict(s.split('-') for s in fname.split('_'))
    return params


def get_extended_L_and_effective(L):
    """
    Parameters
    ----------
    L: array
        Shape (# LoTs, # categories)
        Contains the length of the minimal formula of each category in each LoT
        (note that I only recorded the learning efforts for half of the categories,
        since the NNs show symmetric behaviour for substitution of 0/1 in the input)
        It if -1 for those categories that are not functionally complete
    """
    # add interpretation of each category where 
    # in the input to the neural network 
    # 1 is interpreted as False and 0 as True
    # For instance, category 0000 in category_i
    # would correspond to category 1111 in fliplr(L)
    L_extended = np.concatenate((L,np.fliplr(L)))
    # find the indices where there are minimal formulas length
    # (namely, those that are functionally complete)
    effective_LoTs_indices = np.argwhere(np.all(L_extended!=-1,axis=1)).flatten()
    return L_extended, effective_LoTs_indices


def bitslist_to_binary(list_bits):
    out = 0
    for bit in list_bits:
        out = (out<<1)|int(bit)
    return out

        
def log_mean_exp(A,axis=None):
    A_max = np.max(A, axis=axis, keepdims=True)
    B = (
        np.log(np.mean(np.exp(A - A_max), axis=axis, keepdims=True)) +
        A_max
    )
    return B 


def log_sum_exp(x, axis=None):
    mx = np.max(x, axis=axis, keepdims=True)
    safe = x - mx
    return mx + np.log(np.sum(np.exp(safe), axis=axis, keepdims=True))


def log_normalize(x, axis=None):
    x = np.array(x)
    return x - log_sum_exp(x, axis)


if __name__=='__main__':
    print("hello!")


