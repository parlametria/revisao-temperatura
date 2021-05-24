import numpy as np
import pandas as pd
from google.cloud import storage
import datetime as dt
import matplotlib.pyplot as pl
import sys
import os
import csv
import nltk
import re
import json


### System functions ###

def make_necessary_dirs(filename):
    """
    Create directories in the path to `filename` (str), if necessary.
    """
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


### Exploratory functions ###

def Bold(text):
    """
    Takes a string and returns it bold.
    """
    return '\033[1m'+text+'\033[0m'


def unique(series):
    """
    Takes a pandas series as input and print all unique values, separated by a blue bar.
    """
    u = series.unique()
    try:
        print(Bold(str(len(u)))+': '+'\033[1;34m | \033[0m'.join(sorted(u.astype(str))))
    except:
        print(Bold(str(len(u)))+': '+'\033[1;34m | \033[0m'.join(sorted(u)))

def columns(df):
    """
    Print the number of columns and their names, separated by a blue bar.
    """
    unique(df.columns)
    
def mapUnique(df):
    """
    Takes a pandas dataframe and prints the unique values of all columns and their numbers.
    If the number of unique values is greater than maxItems, only print out a sample.  
    """
    python_version = int(sys.version.split('.')[0])
    maxItems = 20
    
    for c in df.columns.values:
        try:
            u = df[c].unique()
            n = len(u)
        except TypeError:
            u = np.array(['ERROR (probably unhashable type)'])
            n = 'Unknown'
        if python_version == 2:
            isStr = all([isinstance(ui, basestring) for ui in u])
        else:
            isStr = all([isinstance(ui, str) for ui in u])
        print('')
        print(Bold(c+': ')+str(n)+' unique values.')
        
        if n == 'Unknown':
            n = 1
        if n <= maxItems:
            if isStr:
                try:
                    print(',  '.join(np.sort(u)))
                except:
                    print(',  '.join(np.sort(u.astype('unicode'))))
            else:
                try:
                    print(',  '.join(np.sort(u).astype('unicode')))
                except:
                    print(',  '.join(np.sort(u.astype('unicode'))))
        else:
            if isStr:
                try:
                    print(Bold('(sample) ')+',  '.join(np.sort(np.random.choice(u,size=maxItems,replace=False))))
                except:
                    print(Bold('(sample) ')+',  '.join(np.sort(np.random.choice(u.astype('unicode'),size=maxItems,replace=False))))
            else:
                try:
                    print(Bold('(sample) ')+',  '.join(np.sort(np.random.choice(u,size=maxItems,replace=False)).astype('unicode')))
                except:
                    print(Bold('(sample) ')+',  '.join(np.sort(np.random.choice(u.astype('unicode'),size=maxItems,replace=False))))

def checkMissing(df):
    """
    Takes a pandas dataframe and prints out the columns that have missing values.
    """
    colNames = df.columns.values
    print(Bold('Colunas com valores faltantes:'))
    Ntotal = len(df)
    Nmiss  = np.array([float(len(df.loc[df[c].isnull()])) for c in colNames])
    df2    = pd.DataFrame(np.transpose([colNames,[df[c].isnull().any() for c in colNames], Nmiss, np.round(Nmiss/Ntotal*100,2)]),
                     columns=['coluna','missing','N','%'])
    print(df2.loc[df2['missing']==True][['coluna','N','%']])


def freq(series, value):
    """
    Takes a pandas series and a value and returns the fraction of the series that presents a certain value.
    """
    Ntotal = len(series)
    Nsel   = float(len(series.loc[series==value]))
    return Nsel/Ntotal


def one2oneQ(df, col1, col2):
    """
    Check if there is a one-to-one correspondence between two columns in a dataframe.
    """
    n2in1 = df.groupby(col1)[col2].nunique()
    n1in2 = df.groupby(col2)[col1].nunique()
    if len(n2in1)==np.sum(n2in1) and len(n1in2)==np.sum(n1in2):
        return True
    else:
        return False


def one2oneViolations(df, colIndex, colMultiples):
    """
    Returns the unique values in colMultiples for a fixed value in colIndex (only for when the number of unique values is >1).
    """
    return df.groupby(colIndex)[colMultiples].unique().loc[df.groupby(colIndex)[colMultiples].nunique()>1]


### Data processing ###


def pick_representative(elements_list, priority_dict):
    """
    Given a list `elements_list` and a dict `priority_dict` 
    that returns a priority for each element in `elements_list`,
    return the element in `elements_list` with most priority 
    (i.e. lowest value in `priority_dict`).
    """
    sorter = [priority_dict[element] for element in elements_list]
    sorted_cand = [x for _, x in sorted(zip(sorter, elements_list))]
    return sorted_cand[0]


def same_as_previous_entry(df, include_first=False):
    """
    Given a DataFrame `df`, return a boolean Series
    specifying if each entry is the same as the previous
    one. If `include_first` is True, set the first entry 
    (for which there is no previous one) to True.
    """
    
    result = (df == df.shift()).product(axis=1).astype(bool)
    if include_first:
        result.iloc[0] = True
    return result


def compute_differences_for_fixed_id(df, id_columns, sort_columns, diff_columns, sort_index=False):
    """
    Compute the difference between the current and previous entries
    in columns `diff_columns` (str, int or array-like) of DataFrame 
    `df` when it is sorted by columns `sort_columns` (str, int or 
    list), ignoring (i.e. setting to null) when `id_columns` (str, 
    int or list) changes.
    
    Return : Series or DataFrame
        The return type depends on whether `diff_columns` is 
        a list or a str or int.
    """
    
    # Standardize input:
    if type(id_columns) != list:
        id_columns = [id_columns]
    if type(sort_columns) != list:
        sort_columns = [sort_columns]

    # Sort DataFrame:
    sorted_df = df.sort_values(id_columns + sort_columns)
    # Check where IDs change:
    changed_id = ~same_as_previous_entry(sorted_df[id_columns])
    # Compute differences:
    differences = sorted_df[diff_columns] - sorted_df[diff_columns].shift()
    # Ignore changes for different identifiers:
    differences.loc[changed_id] = differences.iloc[0]

    if sort_index:
        differences = differences.sort_index()
    
    return differences


def return_previous_entries(df, id_columns, sort_columns, hist_columns, sort_index=False, name=None):
    """
    Return the previous entries in column `hist_columns` (str) of DataFrame 
    `df` when it is sorted by columns `sort_columns` (str, int or 
    list), ignoring (i.e. setting to null) when `id_columns` (str, 
    int or list) changes.
    
    Return : Series or DataFrame
        The return type depends on whether `hist_columns` is 
        a list or a str or int. The Series name is set to `name`
        unless it is None, in which case it is set to 'prev' + `hist_columns`.
    """

    # Standardize input:
    if type(id_columns) != list:
        id_columns = [id_columns]
    if type(sort_columns) != list:
        sort_columns = [sort_columns]

    # Sort DataFrame by id_columns (to group entries related to the same object) and by sort_columns:
    sorted_df = df.sort_values(id_columns + sort_columns)
    
    # Get previous entries:
    previous_entries = sorted_df[hist_columns].shift()
    # Ignore previous entries when the ID changes:
    changed_id = ~same_as_previous_entry(sorted_df[id_columns])
    previous_entries.loc[changed_id] = previous_entries.iloc[0]
    
    if sort_index:
        previous_entries = previous_entries.sort_index()

    if type(name) != type(None):
        previous_entries.name = name
    else:
        previous_entries.name = 'prev_' + hist_columns
    
    return previous_entries


def generate_label_df(df, agg_cols, count_cols):
    """
    Create a DataFrame in which, for each combination of values present 
    in `df` columns `agg_cols`, create one row for each of the possible 
    combination of all values in columns `count_cols`. That is: the 
    combination of values in `agg_cols` are only those seen in `df`, while 
    any possible combination of them and values in `count_cols` gain one 
    row.
    
    Input
    -----
    
    df : DataFrame
        The DataFrame with columns `agg_cols` and `count_cols` 
        from which the possible values and combinations will be 
        built.
        
    agg_cols : list of str
        Names of the columns to be used to group the data.
        Only the observed combination of their values will 
        be kept in the output.
        
    count_cols : list of str
        Names of the columns for which all the combinations 
        of all available values will be shown in the output, 
        even if such combination is never seen in `df`. This 
        is repeated for each combination of the `agg_cols` 
        values.
        
    Return
    ------
    
    label_df : DataFrame
        A dataframe with columns `agg_cols` and `count_cols`, 
        where the values of the first are combined to reproduced 
        the observed combinations in `df`, and the values of 
        the latter are combined (appear in the same row) in every 
        possible combination between them and with the `agg_cols`.
    """

    # Create DataFrame with columns representing the groups:
    label_df = df.groupby(agg_cols).size().reset_index()[agg_cols]
    label_df.index = pd.Index([0] * len(label_df))

    # Loop over categories to count:
    for col in count_cols:

        # Build series of unique values:
        col_values = df[col].unique()
        count_series = pd.Series(col_values, index=[0] * len(col_values))
        count_series.name = col

        # Join to table of labels:
        label_df = label_df.join(count_series)

    # Reset index:
    label_df = label_df.reset_index(drop=True)
    
    return label_df


def count_occurences_in_groups(df, agg_cols, count_cols, counts_name='counts'):
    """
    Count how many combinations of values in `count_cols` appear in 
    each group identified by `agg_cols`.
    
    Input
    -----
    
    df : DataFrame
        The data containing columns `agg_cols` and `count_cols`, where 
        to count occurences.
        
    agg_cols : list of str
        Names of columns to be used for grouping the data. The counting 
        happens inside each group existent in `df`.
        
    count_cols : list of str
        Columns containing possible values whose all possible combinations 
        should be counted in each group. 
    
    counts_name : str
        Name of the column containing the counts.
        
    Return
    ------
    
    counts_df : DataFrame
        A dataframe with columns `agg_cols`, `count_cols` and `counts_name`
        listing the groups, the combination of values to be counted and 
        the counts.
    """

    # Padroniza input para listas de colunas:
    if type(agg_cols) == str or type(agg_cols) == int:
        agg_cols = [agg_cols]
    if type(count_cols) == str or type(count_cols) == int:
        count_cols = [count_cols]
    
    # Cria grupos para receber a contagem:
    label_df   = generate_label_df(df, agg_cols, count_cols)
    
    # Conta a ocorrência de valores em `count_cols` em cada grupo `agg_cols`:
    obs_counts = df.groupby(agg_cols + count_cols).size()
    obs_counts.name = counts_name

    # Junta os dois para registrar zero ocorrências de `count_cols` em grupos:
    counts_df = label_df.join(obs_counts, on=agg_cols + count_cols)
    counts_df[counts_name] = counts_df[counts_name].fillna(0).astype(int)
    
    return counts_df


def daterange(start_date_str, end_date_str):
    """
    Works the same as python's 'range' but for datetime in days.
    As in 'range', the 'end_date' is omitted.
    """

    start_date = dt.datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date   = dt.datetime.strptime(end_date_str, '%Y-%m-%d')
    
    for n in range(int ((end_date - start_date).days)):
        yield start_date + dt.timedelta(n)

        
class translate_dict(dict):
    """
    A dict that returns the key used if no translation was provided for it.
    """
    def __missing__(self,key):
        return key


def mass_replace(string_list, orig, new):
    """
    For each string in `string_list`, replace each element in `orig` 
    by the corresponding element in `new`.
    
    Input
    -----
    
    string_list : list of str
        The strings in which the replace operation will be performed.
        
    orig : str or list of str
        The substring to be found and replaced in each string in `string_list`,
        or a list of those.
        
    new : str or list of str
        The string that will replace `orig` in `string_list`, or 
        a list of those.
        
    Return
    ------
    
    result : list of str
        List with same length as `string_list`, with all pairs of patterns
        in `zip(orig, new)` replaced, element-wise.
    """
    
    # Standardize input to list of strings:
    if type(orig) == str:
        orig = [orig]
    if type(new) == str:
        new = [new]
    
    # Make sure `orig` and `new` have the same length:
    assert len(orig) == len(new), '`orig` and `new` should have the same length.'
    
    result = string_list
    for o, n in zip(orig, new):
        result = [string.replace(o, n) for string in result]
    return result


def aggregate_strings(df, group_col, text_col, sep='\n\n'):
    """
    Group DataFrame `df` by columns group_col (list or str)
    and concatenate strings in column `text_col` (str) using
    separator `sep` (str). Return a Series whose index is
    given by `group_col`.
    """
    
    agg_series = df.loc[~df[text_col].isnull()].groupby(group_col)[text_col].agg(lambda x: sep.join(x))
    
    return agg_series

        
### String processing functions ###


def text2tag(text):
    """
    Simplify `text` to use it as part os filenames and so on
    (lowercase it, remove accents and spaces).
    """

    # Remove duplicated spaces:
    text = ' '.join(text.split())
    # Transform to tag:
    tag  = re.sub('[\.,;!:\(\)/]', '', remove_accents(text).lower().replace(' ', '_'))
    return tag


def remove_accents(string, i=0):
    """
    Input: string
    
    Returns the same string, but without all portuguese-valid accents.
    """
    accent_list = [('Ç','C'),('Ã','A'),('Á','A'),('À','A'),('Â','A'),('É','E'),('Ê','E'),('Í','I'),('Õ','O'),('Ó','O'),
                   ('Ô','O'),('Ú','U'),('Ü','U'),('ç','c'),('ã','a'),('á','a'),('à','a'),('â','a'),('é','e'),('ê','e'),
                   ('í','i'),('õ','o'),('ó','o'),('ô','o'),('ú','u'),('ü','u'),('È','E'),('Ö','O'),('Ñ','N'),('è','e'),
                   ('ö','o'),('ñ','n'),('Ë','E'),('ë','e'),('Ä','A'),('ä','a')]
    if i >= len(accent_list):
        return string
    else:
        string = string.replace(*accent_list[i])
        return remove_accents(string, i + 1)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', '“!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))

def lowercase(text):
    return text.lower()

def remove_stopwords(text, stopwords=None):

    # Load stopwords if not provided:
    if stopwords == None:
        stopwords = nltk.corpus.stopwords.words('portuguese')
        
    word_list = text.split()
    word_list = [word for word in word_list if not word in set(stopwords)]
    return ' '.join(word_list)
    
def stem_words(text):
    stemmer   = nltk.stem.RSLPStemmer()
    #stemmer   = nltk.stem.PorterStemmer()

    word_list = text.split()
    word_list = [stemmer.stem(word) for word in word_list]
    
    return ' '.join(word_list)

def keep_only_letters(text):
    only_letters = re.sub('[^a-z A-ZÁÂÃÀÉÊÍÔÓÕÚÜÇáâãàéêíóôõúüç]', '', text)
    only_letters = ' '.join(only_letters.split())
    return only_letters


def parse_ptbr_number(string):
    """
    Input: a string representing a float number in Brazilian currency format, e.g.: 1.573.345,98
    
    Returns the corresponding float number.
    """
    number = string
    number = number.replace('.', '').replace(',', '.')
    return float(number)


### Plotting ###

def multiple_bars_plot(df, colors=None, alpha=None, width=0.8, rotation=0, horizontal=False):
    """
    Create a bar plot with bars from different columns 
    in `df` side by side.
    
    Input
    -----
    
    df : DataFrame
        Data to plot as bars. Each row corresponds to a 
        different entry, translating to bar positions, 
        and each column correponds to a different data 
        series, each with a different color. The series
        labels are the column names and the bar location
        labels are the index. The data is plotted in the
        order set in `df`.
        
    colors : list, str or None.
        Colors for the data series (`df` columns).
    
    alpha : list, float or None.
        Transparency of the columns.
        
    width : float
        Total column width formed by all summing the 
        widths of each data series.
        
    rotation : float
        Rotation of the column axis labels, given 
        in degrees.
        
    horizontal : bool
        Whether to use horizontal bar plot or not.
    """
    
    # Count number of columns (associated to bar colors):
    cols   = df.columns
    n_cols = len(cols)
    # Count number of rows (associated to bar positions):
    rows   = df.index
    n_rows = len(rows)

    # Standardize input:
    if type(colors) != list:
        colors = [colors] * n_cols
    if type(alpha) != list:
        alpha = [alpha] * n_cols


    # Set plotting x position:
    ind = np.arange(n_rows)
    # Set width of columns:
    wid = width / n_cols
    
    # Loop over columns:
    for i, col in enumerate(cols):
        # Bar plot:
        if horizontal:
            pl.barh(ind - wid / 2 * (n_cols - 1) + wid * i, df[col], height=wid, color=colors[i], alpha=alpha[i], label=col)
        else:
            pl.bar(ind - wid / 2 * (n_cols - 1) + wid * i, df[col], width=wid, color=colors[i], alpha=alpha[i], label=col)

    # Set tick labels:
    ax = pl.gca()
    if horizontal:
        ax.set_yticks(ind)
        ax.set_yticklabels(rows, rotation=rotation)
    else:
        ax.set_xticks(ind)
        ax.set_xticklabels(rows, rotation=rotation)


### Other functions ###


def begins_with(string, substr):
    """
    Return True if `string` begins with `substr`, else
    return False.
    """
    
    sub_len = len(substr)
    
    if string[:sub_len] == substr:
        return True
    else:
        return False


def sort_list_by_another_list(to_sort, sorter):
    """
    Sort list `to_sort` according to associated elements in list `sorter`.
    """
    return [x for _, x in sorted(zip(sorter, to_sort))]


def digito_verificador_cpf(cpf_first_9):
    """
    Given the first 9 digits of a CPF `cpf_first_9` 
    (int or str), return a string with that CPF's last 
    2 digits. 
    """
    
    # Hard-coded:
    factors = np.arange(11, 1, -1)
    
    # Extract array of digits from CPF:
    cpf = list(str(cpf_first_9).replace('.', '').zfill(9))
    cpf = np.array([int(x) for x in cpf])
    
    # Cálculo do 1o dígito:    
    prod  = factors[1:].dot(cpf)
    resto = prod % 11
    J = 0 if resto < 2 else 11 - resto
    
    # Cálculo do 2o dígito:
    prod  = factors.dot(np.append(cpf, J))
    resto = prod % 11
    K     = 0 if resto < 2 else 11 - resto
    
    return str(10 * J + K)


### TEM BUG!! CORRIGIR! >> o split pode dar errado se o path tiver ../
def saveFigWdate(name):
    """
    Takes a string (a filename with extension) and save the current plot to it, 
    but adding the current date to the filename.
    """
    part = name.split('.')
    t = dt.datetime.now().strftime('%Y-%m-%d')
    filename = part[0]+'_'+t+'.'+part[1]
    pl.savefig(filename, bbox_inches='tight')


def cov2corr(cov):
    """
    Takes a covariance matrix and returns the correlation matrix.
    """
    assert(len(cov) == len(np.transpose(cov))), 'Cov. matrix must be a square matrix.'
    corr = [ [cov[i][j]/np.sqrt(cov[i][i]*cov[j][j]) for i in range(0,len(cov))] for j in range(0,len(cov))]
    return np.array(corr)


def load_json(filename):
    """
    Given a filename (str) of a JSON file, returns a list of dicts (or dict)
    of that JSON.
    """
    with open(filename) as f:
        dictionary = json.load(f)
    return dictionary


### Conexão entre Pandas e Google Big Query ###

def bigquery_to_pandas(query, project='gabinete-compartilhado', credentials_file='/home/skems/gabinete/projetos/keys-configs/gabinete-compartilhado.json'):
    """
    Run a query in Google BigQuery and return its results as a Pandas DataFrame. 

    Input
    -----

    query : str
        The query to run in BigQuery, in standard SQL language.
    project : str
        
    
    Given a string 'query' with a query for Google BigQuery, returns a Pandas 
    dataframe with the results; The path to Google credentials and the name 
    of the Google project are hard-coded.
    """

    import google.auth
    import os

    # Set authorization to access GBQ and gDrive:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_file

    
    credentials, project = google.auth.default(scopes=[
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/bigquery',
    ])
    
    return pd.read_gbq(query, project_id=project, dialect='standard', credentials=credentials)


def load_data_from_local_or_bigquery(query, filename, force_bigquery=False, save_data=True, 
                                     project='gabinete-compartilhado', 
                                     credentials_file='/home/skems/gabinete/projetos/keys-configs/gabinete-compartilhado.json',
                                     low_memory=False):
    """
    Loads data from local file if available or download it from BigQuery otherwise.
    
    
    Input
    -----
    
    query : str
        The query to run in BigQuery.
    
    filename : str
        The path to the file where to save the downloaded data and from where to load it.
        
    force_bigquery : bool (default False)
        Whether to download data from BigQuery even if the local file exists.
        
    save_data : bool (default True)
        Wheter to save downloaded data to local file or not.
        
    project : str (default 'gabinete-compartilhado')
        The GCP project where to run BigQuery.
        
    credentials_file : str (default path to 'gabinete-compartilhado.json')
        The path to the JSON file containing the credentials used to access GCP.
        
    low_memory : bool (default False)
        Whether or not to avoid reading all the data to define the data types
        when loading data from a local file.

    Returns
    -------
    
    df : Pandas DataFrame
        The data either loaded from `filename` or retrieved through `query`.
    """
    
    # Download data from BigQuery and save it to local file:
    if os.path.isfile(filename) == False or force_bigquery == True:
        print('Loading data from BigQuery...')
        df = bigquery_to_pandas(query, project, credentials_file)
        if save_data:
            print('Saving data to local file...')
            df.to_csv(filename, quoting=csv.QUOTE_ALL, index=False)
    
    # Load data from local file:
    else:
        print('Loading data from local file...')
        df = pd.read_csv(filename, low_memory=low_memory)
        
    return df


def upload_to_storage_gcp(bucket, key, data):
    """
    Given a data bucket (e.g. 'brutos-publicos') a key (e.g. 
    'executivo/federal/servidores/data/201901_Cadastro.csv'), 
    and 'data' (a string with all the data), write to GCP storage.
    """
    storage_client = storage.Client(project='gabinete-compartilhado')

    bucket = storage_client.get_bucket(bucket)
    blob = bucket.blob(key)

    blob.upload_from_string(data)


def gcp_key(prefix, filename, date_partitioned=True):
    """
    Return a string with the key for the object in the cloud 
    that will contain data from the file `filename` (str).
    Use the `prefix` and use hive partitioning if requested.
    """
    # Get file name (no dirs):
    basename = os.path.basename(filename)
    
    # Make sure prefix ends in slash:
    if prefix[-1] != '/':
        prefix = prefix + '/'   
        
    # Add hive partitioning if requested:
    if date_partitioned:
        key = prefix + 'data=' + basename[:4] + '-' + basename[4:6] + '-01/' + basename
    else:
        key = prefix + basename
        
    return key


def upload_single_file(filename, bucket, prefix, date_partitioned=True, verbose=False, credentials='/home/skems/gabinete/projetos/keys-configs/gabinete-compartilhado.json'):
    """
    Upload file `filename` (str) to Google Storage's `bucket`, defining the key with a 
    `prefix` (str), a hive date partitioning (if `date_partitioned` is True) derived from 
    the date in `filename`. Use the JSON file with path `credentials` as GCP credentials.
    """
    
    # Set object key (where to find the file in GCP):
    if verbose:
        print('Generating Storage key...')
    key = gcp_key(prefix, filename, date_partitioned)
    
    # Read file:
    if verbose:
        print('Reading file...')
    with open(filename, 'r') as f:
        text = f.read()
    
    # Upload file:
    if verbose:
        print('Uploading data to GCP...')
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials 
    upload_to_storage_gcp(bucket, key, text)
