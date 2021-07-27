
### Functions for machine learning ###

import numpy as np
import pandas as pd
from zlib import crc32
from sklearn.base import BaseEstimator, TransformerMixin, clone


###########################################
### Splitting datasets into random sets ###
###########################################

def shuffled_pos(length, seed):
    """
    Return indices from 0 to `length` - 1 in a shuffled state, given random `seed`.
    """
    return np.random.RandomState(seed=seed).permutation(length)


def random_index_sets(size, set_fracs, seed):
    """
    Return sets of random indices (from 0 to `size` - 1) with lengths 
    given by ~ `size` * `set_fracs`.
    
    
    Input
    -----
    
    size : int
        The size of the index list to split into sets.
        
    set_fracs : iterable
        The fractions of the list of indices that each index set 
        should contain. 
    
    seed : int
        The seed for the random number generator.
        
        
    Returns
    -------
    
    indices : tuple of arrays
        The indices for each set.
    """
    
    assert np.isclose(np.sum(set_fracs), 1), '`set_fracs` should add up to one.'
    
    # Create randomized list of indices:
    shuffled_indices = shuffled_pos(size, seed)
    
    
    indices   = []
    set_start = [0]
    # Determine the sizes of the sets:
    set_sizes = [round(size * f) for f in set_fracs]
    set_sizes[0] = size - sum(set_sizes[1:])
    assert np.sum(set_sizes) == size, 'Set sizes should add up to total size.'
    
    for i in range(0, len(set_fracs) - 1):
        # Select indices for a set:
        set_start.append(set_start[i] + set_sizes[i])
        set_indices = shuffled_indices[set_start[i]:set_start[i + 1]]
        indices.append(set_indices)
        assert len(indices[i]) == len(set(indices[i])), 'There are repeating indices in a set.'
        
    # Select the indices for the last set:
    indices.append(shuffled_indices[set_start[-1]:])
    assert len(set(np.concatenate(indices))) == sum([len(i) for i in indices]), \
    'There are common indices between sets.'
    
    return tuple(indices)


def random_set_split(df, set_fracs, seed):
    """
    Split a DataFrame into randomly selected disjoint and complete sets.
    
    
    Input
    -----
    
    df : Pandas DataFrame
        The dataframe to split into a complete and disjoint set of sub-sets.
        
    set_fracs : array-like
        The fraction of `df` that should be put into each set. The length of 
        `set_fracs` determines the number of sub-sets to create.
    
    seed : int
        The seed for the random number generator used to split `df`.
        
    
    Returns
    -------
    
    A tuple of DataFrames, one for each fraction in `set_fracs`, in that order.
    """
    # Get positional indices for each set:
    sets_idx = random_index_sets(len(df), set_fracs, seed)
    
    return tuple(df.iloc[idx] for idx in sets_idx)


def hash_string(string, prefix=''):
    """
    Takes a `string` as input, remove `prefix` from it and turns it into a hash.
    """
    name   = string.replace(prefix, '')
    return crc32(bytes(name, 'utf-8'))


def test_set_check_by_string(string, test_frac, prefix=''):
    """
    Returns a boolean array saying if the data identified by `string` belongs to the test set or not.
    
    Input
    -----
    
    string : str
        The string that uniquely identifies an example.
    
    test_frac : float
        The fraction of the complete dataset that should go to the test set (0 to 1).
        
    prefix : str (default '')
        A substring to remove from `string` before deciding where to place the example.
        
        
    Returns
    -------
    
    A bool number saying if the example belongs to the test set.
    """
    return hash_string(string, prefix) & 0xffffffff < test_frac * 2**32


def train_test_split_by_string(df, test_frac, col, prefix=''):
    """
    Split a DataFrame `df` into train and test sets based on string hashing.
    
    Input
    -----
    
    df : Pandas DataFrame
        The data to split.
        
    test_frac : float
        The fraction of `df` that should go to the test set (0 to 1).

    col : str or int
        The name of the `df` column to use as identifier (to be hashed).
        
    prefix : str (default '')
        A substring to remove from the rows in column `col` of `df` 
        before deciding where to place the example.
        
    Returns
    -------
    
    The train and the test sets (Pandas DataFrames).
    """
    ids = df[col]
    in_test_set = ids.apply(lambda s: test_set_check_by_string(s, test_frac, prefix))
    return df.loc[~in_test_set], df.loc[in_test_set]


def train_test_split_by_date(df, date_col, test_min_date):
    """
    Given a DataFrame `df` with a date column `date_col`, split it into 
    two disjoint DataFrames with data previous to date `test_min_date` 
    and data equal to or later than `test_min_date`. These are return 
    is this order.
    """
    train = df.loc[df[date_col] <  test_min_date]
    test  = df.loc[df[date_col] >= test_min_date]
    return train, test


def train_val_test_split_by_date_n_string(df, date_col, min_test_date, str_col, prefix=''):
    """
    Split a DataFrame `df` into 3 disjoint sets (train, validation and test):
    The first (train) contains all data with dates stored in column 
    `date_col` that are less than `min_test_date`. The last two sets (validation 
    and test) contain data from period equal to or later than `min_test_date`.
    These two sets have approximatelly the same size and are selected by 
    hashing strings in column `str_col`; these strings can have a substring
    `prefix` removed from them before hashing.    
    """
    # Quebra a base em passado (train) e futuro (test e validação):
    train, val_test = train_test_split_by_date(df, date_col, min_test_date)
    n_train    = len(train)
    n_val_test = len(val_test)

    # Quebra o futuro em test e validação usando string hashing:
    val, test = train_test_split_by_string(val_test, 0.5, str_col, prefix)
    n_val     = len(val)
    n_test    = len(test)

    print('# train:     ', n_train)
    print('# validation: {:d}  ({:.1f}%)'.format(n_val, n_val / (n_val + n_train) * 100))
    print('# test:       {:d}  ({:.1f}%)'.format(n_test, n_test / (n_test + n_train) * 100))   
    
    return train, val, test


def Xy_split(df, y_col):
    """
    Given a Pandas DataFrame `df` and a column name `y_col` (str), returns:
    - The features X (`df` without column `y_col`);
    - The target y (`df[y_col]`)  
    """
    out_df = df.copy()
    
    X = out_df.drop(y_col, axis=1)
    y = out_df[y_col]
    
    return X, y


def stratified_split(df, labels, test_size, random_state):
    """
    Return `df_train`, `df_test`, `labels_train`, `labels_test` by 
    splitting `df` (DataFrame) and `labels` (Series) into two sets 
    using Stratified sampling.
    """
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = tuple(strat_split.split(df, labels))[0]

    df_train = df.iloc[train_idx]
    df_test  = df.iloc[test_idx]
    
    labels_train = labels.iloc[train_idx]
    labels_test  = labels.iloc[test_idx]
    
    return df_train, df_test, labels_train, labels_test


######################
### Baseline model ###
######################


class RandomPicker(BaseEstimator):
    """
    A model whose predictions are random.

    Input
    -----

    follow_frequency : bool (default False)
        If True, randomly samples from the y sample used to train,
        so the frequency of each value follows the training sample.
        If False, sample each value uniformly.
    """    
    def __init__(self, follow_frequency=False):
        self.follow_frequency = follow_frequency
    
    def fit(self, X, y=None):
        self.y_sample = y
        return self
    
    def predict(self, X):
        n = X.shape[0]
        
        if self.follow_frequency:
            choices = self.y_sample
            return np.random.choice(choices, n)
        
        else:
            choices = np.unique(self.y_sample)
            return np.random.choice(choices, n)
