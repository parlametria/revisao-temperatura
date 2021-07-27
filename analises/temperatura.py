import pandas as pd
import numpy as np
import os
from termcolor import colored
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import utils as xu
import mltools as ml


def formatted_print(text, color=None, bold=False, end='\n'):
    """
    Print a string `text` using `color` (str) with a bold
    weight in `bold` (bool) is True, ends the line with `end`
    (str).
    """
    text = str(text)
    
    if bold:
        print(xu.Bold(colored(text, color)), end=end)
    else:
        print(colored(text, color), end=end)


def print_one_event(row_series):
    """
    Pretty print info about an entry in a proposição's tramitação.
    """
    try:
        data = row_series['data'].strftime('%Y-%m-%d %H:%M:%S')
    except:
        data = row_series['data']
        
    # Instante:
    formatted_print('({})'.format(row_series['sequencia']), 'blue', bold=True, end=' ')
    formatted_print(data, bold=True, end = ' - ')
    
    # Local:
    formatted_print('[{}] '.format(row_series['casa']), 'green', bold=True, end='')
    formatted_print('{} >> '.format(row_series['origem_tramitacao_local_nome_casa_local']), 'green', bold=True, end='')
    formatted_print('[{}]'.format(row_series['local']), 'green', bold=True, end=' ')
    formatted_print(row_series['sigla_local'], 'green', bold=True, end=' : ')
    
    # Situação:
    formatted_print(row_series['global'], 'red', end=' ')
    formatted_print('<{}> - '.format(row_series['fase']), 'red', end='')
    #formatted_print('[{}]'.format(row_series['situacao_descricao_situacao']), 'red', end=' ')
    formatted_print('{}'.format(row_series['descricao_situacao']), 'red', end=' ! ')
    
    # Ação legislativa:
    formatted_print('[{}]'.format(row_series['evento']), 'blue', bold=True, end=' ')
    formatted_print('{}'.format(row_series['titulo_evento']), 'blue', bold=True, end=' - ')
    
    # Outros:
    formatted_print('{}'.format(row_series['tipo_documento']), end=' ')
    formatted_print('{}'.format(row_series['nivel']), end=' ')
    formatted_print('({})'.format(row_series['status']), bold=True)

    print(row_series['texto_tramitacao'])


def print_header(df):
    """
    Print fixed info about a proposição.
    """
    pid_casa  = df[['id_ext', 'casa']].drop_duplicates()
    
    for i in range(len(pid_casa)):
        row = pid_casa.iloc[i]
        formatted_print(row['id_ext'], bold=True, end=' ')
        formatted_print('({})'.format(row['casa']), bold=True, end=' --> ')
        if row['casa'] == 'camara':
            formatted_print('https://www.camara.leg.br/proposicoesWeb/fichadetramitacao?idProposicao={}'.format(row['id_ext']))
        else:
            formatted_print('https://www25.senado.leg.br/web/atividade/materias/-/materia/{}'.format(row['id_ext']))
            
    # Print field names:
    col_names = pd.Series(df.columns, index=df.columns)
    print_one_event(col_names)

    
def print_tramitacao(tram):
    """
    Pretty print tramitação of one proposição 
    listed in DataFrame `tram`.
    """
    # Print tramitação:
    print_header(tram)
    print('')
    for i in range(len(tram)):
        row = tram.iloc[i]
        print_one_event(row)
        print('')


def build_leggo_id_unifier(proposicoes_df):
    """
    Identify proposições that are the same but have 
    different Leggo IDs through the use of their 
    external IDs and create a translating function
    that can group the Leggo IDs into a single one.
    
    Input : proposicoes_df : DataFrame
        A dataframe with the columns 'íd_ext' and 
        'id_leggo'.
    """
    
    # Count the repetitions of external IDs:
    id_ext_entries = proposicoes_df['id_ext'].value_counts()
    # Select the ones that repeat:
    repeated_id_ext = id_ext_entries.loc[id_ext_entries > 1].index
    # Verifica que um ID externo se repete no máximo duas vezes:
    assert id_ext_entries.max() <=2

    # For each one, get their Leggo IDs:
    id_leggo_arrays = proposicoes_df.loc[proposicoes_df['id_ext'].isin(repeated_id_ext)].groupby('id_ext')['id_leggo'].unique()
    # Create a dict that translates one Leggo ID to another:
    leggo_id_unifier = dict(zip(id_leggo_arrays.apply(lambda x: x[0]), id_leggo_arrays.apply(lambda x: x[1])))

    return xu.translate_dict(leggo_id_unifier)


def select_one_tramitacao(trams_df, unique_id_leggo, na_position='last'):
    """
    Select from `trams_df`  (DataFrame all tramitações of 
    many proposições) the tramitação of a single proposição 
    identified by `unique_id_leggo` (str) column.
    
    Return a DataFrame of tramitações in chronological 
    order.
    """
    tramitacao = trams_df.loc[trams_df['unique_id_leggo'] == unique_id_leggo].sort_values(['data', 'sequencia'], na_position=na_position)
    
    return tramitacao


def get_time_interval(tramitacoes, id_columns='unique_id_leggo', sort_columns=['data', 'sequencia'], diff_columns='data', name='delta_days'):
    """
    Compute the time interval (in days) between entries in 
    DataFrame `tramitacoes` of the same ID.
    
    Return : Series
        A Series with the number of days between the 
        current and previous entries in `tramitacoes`.
    """
    
    # Compute time difference between events:
    delta_days = xu.compute_differences_for_fixed_id(tramitacoes, id_columns, sort_columns, diff_columns)
    delta_days = delta_days.dt.total_seconds() / 3600 / 24
    
    delta_days.name = name
    
    return delta_days


def select_relevant_events(tramitacoes, relevant_markers):
    """
    Select entries from `tramitacoes` (DataFrame) that 
    have at least one non-null value  in the columns 
    listed in `relevant_markers` (str, int or list).
    
    Return a DataFrame.
    """
    
    sel_tramitacoes = tramitacoes.loc[~tramitacoes[relevant_markers].isnull().product(axis=1).astype(bool)]
    
    return sel_tramitacoes


def fillna_fixed_tram(df_trams):
    """
    Fill missing values if a tramitações DataFrame
    `df_trams` where every proposição has the same 
    number of entries. 
    
    PS: `df` must be sorted by prop. id and date 
    (descending) and missing values must be placed 
    first.
    """
    
    for col in df_trams.columns:
        
        if col in ['casa', 'data', 'acao_futura', 'delta_aval']:
            df_trams[col].fillna(method='bfill', inplace=True)
            
        if col in ['local', 'sigla_local', 'descricao_situacao', 'evento', 'titulo_evento']:
            df_trams[col].fillna('XXXXXX', inplace=True)
            
        if col in ['nivel']:
            df_trams[col].fillna(4, inplace=True)
            
        if col in ['sequencia', 'delta_days', 'acoes_vazias']:
            df_trams[col].fillna(0, inplace=True)
            
    return df_trams


def fillna_relevant_tram(df_trams, event_identifier):
    """
    Fill missing values in `df`, a DataFrame with
    tramitações of many proposições.
    
    PS: Currently it also eliminates missing dates 
    in 'data' column. This may be improved by using 
    information from the 'sequencia' column.
    """
    
    # Fill missing event identifiers:
    for col in event_identifier:
        df_trams[col].fillna('XXXXXX', inplace=True)

    # Set missing time intervals (should always be for the first action of each proposition) to zero:
    df_trams['delta_days'].fillna(0, inplace=True)

    # Remove entries without dates:
    df_trams = df_trams.loc[~df_trams['data'].isnull()]    

    # Não deveriam haver mais valores faltantes:
    assert df_trams.isna().sum().sum() == 0
    
    return df_trams


def crop_and_pad_trams(df_trams, last_n_actions, id_cols, sorter_cols):
    """
    Standardize the number of actions (entries) per 
    proposição by cropping and padding actions.
    
    Input
    -----
    
    df_trams : DataFrame
        Description of tramitações of many 
        proposições.
        
    last_n_actions : int
        Number of the last actions (entries) for 
        each proposição to keep in the dataset.
    
    id_cols : list of str
        Columns used to identify a proposição
        (e.g. [unique_id_leggo]).
    
    sorter_cols : list of str
        Columns used to sort tramitações for
        each proposição (e.g. ['data', 'sequencia']).
        
    Return
    ------
    
    fixed_tramitacoes : DataFrame
        Standardized tramitações, where all proposições 
        have the same number of actions. If the original 
        number of actions is lower than `last_n_actions`
        for a certain proposição, pad (backward fill) 
        the tramitações.
    """
    
    # Não deveriam haver valores faltantes:
    assert df_trams.isna().sum().sum() == 0
    
    # Seleciona as últimas `last_n_actions` de cada proposição: 
    last_tramitacoes = df_trams.sort_values(sorter_cols).groupby(id_cols, sort=True).tail(last_n_actions)

    # Contabiliza número que falta de ações para cada proposição:
    n_empty_actions = last_n_actions - last_tramitacoes.groupby(id_cols).size()

    # Cria ações vazias para proposições com poucas ações:
    empty_series = n_empty_actions.loc[n_empty_actions > 0].map(lambda n: [n] * n).explode()
    empty_series.name = 'acoes_vazias'

    # Coloca ações vazias na base para padronizar número de ações:
    fixed_tramitacoes = pd.concat([empty_series.reset_index(), last_tramitacoes], ignore_index=True).sort_values(id_cols + sorter_cols, na_position='first')

    # Define valores para ações vazias:
    fixed_tramitacoes = fillna_fixed_tram(fixed_tramitacoes)
    
    # Revert int column types to int:
    for col in ['nivel', 'sequencia', 'acao_futura']:
        fixed_tramitacoes[col] = fixed_tramitacoes[col].astype(int)
    
    return fixed_tramitacoes


def std_comissao_especial(series):
    """
    Return `series` with comissões especiais
    (e.g. 'MPV34867') replaced by 'Comissão Especial'.
    """
    return series.str.replace('^(?:MPV|PEC|PL|PLP)\d{4,7}', 'Comissão Especial')


def log10_one_plus(x):
    """
    Return log10(1 + x) for `x` (array-like).
    """
    return np.log10(1 + x)


def unstack_snapshots(X, n_snapshots):
    """
    Take a feature matrix `X` (array) where there are 
    `n_snapshots` (int) rows for each example, each 
    row representing a different (time) snapshot of 
    the example, and reshape it so all information 
    about each example rests on the same row.
    """
    
    n_features = X.shape[1]
    return X.reshape((-1, n_features * n_snapshots))


class UnstackSnapshots(BaseEstimator, TransformerMixin):
    """
    Take a feature matrix `X` (array) where there are 
    `n_snapshots` (int) rows for each example, each 
    row representing a different (time) snapshot of 
    the example, and reshape it so all information 
    about each example rests on the same row.
    """
    
    def __init__(self, n_snapshots):
        self.n_snapshots = n_snapshots
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return unstack_snapshots(X, self.n_snapshots)
    

def add_lag_data(df, id_columns, sort_columns, lag_columns, shift, suffix):
    """
    Join to `df` data from columns `lag_columns` (list)
    of `df` (DataFrame) with the same ID (given in 
    `id_columns`, list) but with a lag of `shift` (int) 
    entries when `df` is sorted by `sort_columns`. This
    lag data receives the `suffix` in their column names.
    """
    
    # Security checks:
    assert shift < 0, 'This function assumes that `shift < 0`.'
    
    # Sort data:
    sorted_df = df.sort_values(id_columns + sort_columns)
    # Check when id has changed to avoid mixing data from different ids: 
    changed_id = ~xu.same_as_previous_entry(sorted_df[id_columns], shift=shift)
    
    # Get shifted (lag) data:
    shifted = sorted_df[lag_columns].shift(shift)
    # Ignore shifted values when the id is different:
    shifted.loc[changed_id] = shifted.iloc[-1]
    # Rename columns in shifted data:
    lag_names = [name + suffix for name in lag_columns]
    shifted.rename(dict(zip(lag_columns, lag_names)), axis=1, inplace=True)

    # Join to original data:
    joined = df.join(shifted)
    
    return joined


def create_future_event_extension(future_event, max_date):
    """
    Given a DataFrame `future_event` with columns
    'data_aval' and 'acao_futura', return a 
    DataFrame with a column 'data_aval' containing:
    all dates from one day after the last in 
    `future_event` up to `max_date` (str in format
    '%Y-%m-%d'); and zeros under column 
    'acao_futura'.
    """
    
    df = pd.DataFrame()
    # Build assessment dates from where the data stopped:
    df['data_aval'] = pd.date_range(future_event['data_aval'].max() + pd.DateOffset(days=1), pd.to_datetime(max_date, utc=True))
    # Set no relevant events for extension dates:
    df['acao_futura'] = 0

    return df


def extend_future_events(future_event, max_date):
    """
    Given a DataFrame `future_event` with 
    daily entries stating if a relevant event 
    happened in the future, extend this entries
    up to `max_date` (str in format '%Y-%m-%d')
    with no relevant events.
    """
    
    extension = create_future_event_extension(future_event, max_date)
    extended  = pd.concat([future_event, extension], ignore_index=True)
    
    return extended


def compute_undersample_size(n_entries, undersample_factor):
    """
    Given a number of elements `n_entries` (int),
    return the number of elements that can be 
    selected once every `undersample_factor` (int),
    plus one. 
    
    Another way to explain this is: given a set of 
    elements `n_entries`, return the number of bin 
    edges that would split the entries into 
    bins contaning `undersample_factor` entries.
    """
    
    return int(1 + int(n_entries / undersample_factor))


def build_event_detector(sorter_cols, n_days_ahead, avg_aval_timestep, max_date, max_nivel=1, id_col='unique_id_leggo', random_state=None):
    """
    Build a function that can be applied to a 
    DataFrame `df` containing the history of events for 
    a single proposição to return a randomly sampled 
    DataFrame covering the time range from proposição's 
    first event up to the specified maximum date  
    answering, for each day, whether something 
    important happened on the next `n_days_ahead` 
    (int) days.
    
    Input
    -----
    
    sorter_cols : list or str
        Column names used to sort columns in 
        chronological order.
    
    n_days_ahead : int
        Number of days after each date in the returned 
        DataFrame to look for relevant events.
    
    avg_aval_timestep : int
        Average number of days between each evaluation 
        date (i.e. the dates for which we ask if something 
        relevant happened in the next `n_days_ahead` days).
        
    max_date : str
        Last assessment date, in format '%Y-%m-%d'.
        This date should not exceed the date the data
        was collected.
    
    max_nivel : float
        Maximum event level (from 1 to 4) that should
        me considered relevant.
        
    id_col : str
        Column that identifies a single proposição.
        Only used for security check.
        
    random_state : int or None
        Seed for random number generator. Set to None
        for non-reproducible run.
    """
    
    def event_on_next_days(df):
        """
        Given a DataFRame `df` with the history of events for 
        a single proposição, create a randomly sampled DataFrame 
        covering the proposição's time range answering, for
        each day, whether something important happened on the
        next n days.        
        """
        
        # Security checks:
        assert df[id_col].nunique() == 1, "This function should only be applied to a single '{}'.".format(id_col)
        
        # Resample events to show the best rank that happened on the next day, for every day: 
        events = df.sort_values(sorter_cols).set_index('data')
        nivel_by_day = events['nivel'].resample('1d').min()
        nivel_by_next_day = nivel_by_day.shift(-1)
    
        # Check the best rank that appeared on the next `n_days_ahead` days, for every day:
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=n_days_ahead)
        event = (nivel_by_next_day.rolling(window=indexer, min_periods=1).min() <= max_nivel).astype(int)
        
        # Remove entries with missing future information:
        # (NOT ANYMORE: NOW WE ADD ASSESSMENT DATES UP TO max_date)
        #future_event = event.iloc[:-n_days_ahead]
        future_event = event
        
        # Organize series:
        future_event.name = 'acao_futura'
        future_event.index.name = 'data_aval'
        future_event = future_event.reset_index()
        
        # Fill assessment dates up to max date with no relevant events:
        future_event = extend_future_events(future_event, max_date)
        
        # Undersample evaluations dates:
        n_data_aval = len(future_event)
        if n_data_aval > 0:
            n_avals = compute_undersample_size(n_data_aval, avg_aval_timestep)
            future_event = future_event.sample(n_avals, random_state=random_state).sort_values('data_aval')

        return future_event
    
    return event_on_next_days 


def augment_and_create_target_subset(df_trams, selected_ids, next_days, avg_aval_timestep, max_date, max_nivel, sorter_cols, id_col='unique_id_leggo', random_state=None):
    """
    For each tramitação history in `df_trams` (DataFrame) of
    proposições listed in `selected_ids` (array-like),
    replicate the history for different assessing dates 
    (up to `max_date`) and add a column that tells if a 
    relevant event (i.e. with 'nivel' column equal to or 
    less than `max_nivel`, float) happened in the 
    `next_days` (int) from the assessing date. 
    
    Proposições are identified by `id_col` (str) and sorted 
    by `sorter_cols` (list of str).
    """
    
    # Select subset of proposições:
    sel = df_trams.loc[df_trams[id_col].isin(selected_ids)]

    # Create y for the next n days, for each assessing date and proposição:
    event_detector = build_event_detector(sorter_cols, next_days, avg_aval_timestep, max_date, max_nivel, id_col, random_state)
    acoes_futuras = sel.groupby(id_col).apply(event_detector)
    acoes_futuras = acoes_futuras.droplevel(1) 
    
    # Join y with the historical:
    sel_com_acoes_futuras = sel.join(acoes_futuras, on=id_col)
    # (some proposições with short history won't have any future measurements; the selection below takes care of removing these proposições)
    
    # Select only tramitações prior to assessment date:
    historico_e_futuro = sel_com_acoes_futuras.loc[(sel_com_acoes_futuras['data'].dt.date) <= sel_com_acoes_futuras['data_aval'].dt.date].sort_values([id_col, 'data_aval'] + sorter_cols)
    historico_e_futuro['acao_futura'] = historico_e_futuro['acao_futura'].astype(int) 
    
    # Compute interval (days) between assessment and event date:
    historico_e_futuro['delta_aval'] = (historico_e_futuro['data_aval'] - historico_e_futuro['data']).dt.total_seconds() / 3600 / 24
    
    return historico_e_futuro


def augment_create_target_and_crop(df_trams, selected_ids, next_days, avg_aval_timestep, max_date, max_nivel, last_n_actions, sorter_cols, random_state=1232234):
    """
    For each tramitação history in `df_trams` (DataFrame) of
    proposições listed in `selected_ids` (array-like),
    replicate the history for different assessing dates 
    (up to `max_date`) and add a column ('acao_futura') that 
    tells if a relevant event (i.e. with 'nivel' column equal 
    to or less than `max_nivel`, float) happened in the 
    `next_days` (int) from the assessing date. 
    
    Then, standardize the number of actions (entries) per 
    proposição and assessment date to `last_n_actions` (int) 
    by cropping and padding actions.
    
    Events are sorted by `sorter_cols` (list of str).
    """
    
    # Augment examples with different assessment dates and add target:
    historico_e_futuro = augment_and_create_target_subset(df_trams, selected_ids, next_days, avg_aval_timestep, max_date, max_nivel, sorter_cols, 'unique_id_leggo', random_state)

    # Standardize the number of actions per proposição:
    fixed_tramitacoes = crop_and_pad_trams(historico_e_futuro, last_n_actions, ['unique_id_leggo', 'data_aval'], sorter_cols)
    
    return fixed_tramitacoes


def select_tipo_proposicao(df_trams, sigla_tipo):
    """
    Select proposições of a certain type. 
    This is relevant since different proposições
    may follow different tramitações.
    
    Types with distinct tramitações:
    (MPV, PLV), PEC, PDS, PLN, PLP, (PL, PLS, PLC)
    """
    # Standardize input:
    if type(sigla_tipo) == str:
        sigla_tipo = [sigla_tipo]
    
    sel = df_trams.loc[df_trams['sigla_tipo'].isin(sigla_tipo)]
    
    return sel


def check_missing_prop_types(df_trams, casa, force_bigquery=False):
    """
    Make sure that all proposições in `df_trams` from 
    `casa` (str, 'camara' or 'senado') without an assigned 
    type are MPVs. It does not return anything. 
    """
    
    # Hard-coded:
    camara_query_template = """
    SELECT id, siglaTipo AS sigla_tipo 
    FROM `gabinete-compartilhado.camara_v2_processed.proposicoes_cleaned` 
    WHERE id IN ({})
    """
    senado_query_template = """
    SELECT Codigo_Materia AS id, Sigla_Subtipo_Materia AS sigla_tipo 
    FROM `gabinete-compartilhado.senado_processed.proposicoes_cleaned` 
    WHERE Codigo_Materia IN ({})
    """
    
    # Seleciona proposições da câmara sem regitro de tipo:
    no_sigla_tipo = df_trams.loc[(df_trams['casa'] == casa) & df_trams['sigla_tipo'].isnull(), 'id_ext'].drop_duplicates()

    # Carrega lista de tipos para essas proposições:
    if casa == 'camara':
        query = camara_query_template.format(','.join(no_sigla_tipo.astype(str)))
    elif casa == 'senado':
        query = senado_query_template.format(','.join(no_sigla_tipo.astype(str)))
    else:
        raise Exception("Unknown `casa` '{}'.".format(casa))        
    tipo_df = xu.load_data_from_local_or_bigquery(query, '../dados/sigla_tipo_proposicoes_{}.csv'.format(casa), force_bigquery=force_bigquery)

    # Confere que a lista contém todas as proposições que faltam:
    assert set(tipo_df['id']) == set(no_sigla_tipo), 'As proposições sem tipo não são as mesmas da lista com tipos: talvez você deva rodar esta função com `force_bigquery = True`.).'

    # Confirma que as proposições sem tipo são MPVs:
    missing_tipos = tipo_df['sigla_tipo'].unique()
    assert len(missing_tipos) == 1
    assert missing_tipos[0] == 'MPV'
    
    print('Todas as proposições sem tipo são MPVs.')

    
def save_trams_by_tipo(df_trams, sigla_tipo, prefix, index=False):
    """
    Select tramitações de proposições from `df_trams` 
    (DataFrame) as belonging to a certain list of 
    types `sigla_tipo` (str or list of str, e.g. ['MPV', 
    'PLV']) and save it to a CSV file with a given 
    `prefix` (str) followed by the `sigla_tipos` and 
    the extension `.csv`.
    """
    
    # Standardize input:
    if type(sigla_tipo) == str:
        sigla_tipo = [sigla_tipo]

    # Select proposições by tipo:
    sel = select_tipo_proposicao(df_trams, sigla_tipo)
    
    # Save to file
    filename = prefix + '_'.join(sigla_tipo) + '.csv'
    sel.to_csv(filename, index=index)

    
def sequence_dataset_split(df, test_frac, example_id_col, random_state=98128123):
    """
    Given a DataFrame of examples `df` where each example,
    uniquely identified by the value on column `example_id_col`,
    is composed of several rows, split the `df` into 
    train, validation and test sets.
    
    The first split selects a fraction of `test_frac` (float)
    examples to be a test set using a hash function, so later
    additions to `df` do not change the former member of the 
    test set. The remaining data is called the 'build' set.
    
    The second split randomly splits the examples in the build 
    set into train and validation set given a seed 
    `random_state` (int), assigning a fraction `test_frac` of 
    the build set to the validation set.
    
    Return
    ------
    
    train_df, val_df, test_df : DataFrames
    """
    
    # Get a testing sample:
    # (that never overlaps with the training/validation samples, even if new examples are added to `df`):
    build_df, test_df = ml.train_test_split_by_string(df, test_frac, example_id_col)
    
    # For the non-test set, get list of IDs and its size:
    n_build = build_df[example_id_col].nunique()
    build_unique_ids = build_df[example_id_col].unique()
    
    # Genetate random positions in the list of IDs:
    build_shuffled_pos = ml.shuffled_pos(n_build, random_state)
    train_id_pos = build_shuffled_pos[ int(test_frac * n_build):]
    val_id_pos   = build_shuffled_pos[:int(test_frac * n_build) ]
    
    # Use random positions to select IDs for train and val. sets:
    train_unique_ids = build_unique_ids[train_id_pos]
    val_unique_ids   = build_unique_ids[val_id_pos]
    
    # Use the randomly selected IDS to split non-test into training and validation samples:
    train_df = build_df.loc[build_df[example_id_col].isin(train_unique_ids)]
    val_df   = build_df.loc[build_df[example_id_col].isin(val_unique_ids)]
    
    return train_df, val_df, test_df


def get_target_from_sequence_set(df_trams, example_id_cols, target_col):
    """
    Given a DataFrame `df`  where each example has
    features spread among rows, return a single 
    target value for each example. 
    
    This assumes that all rows from the same 
    example have the same target value.
    """
    
    return df_trams.drop_duplicates(subset=example_id_cols)[target_col]


def binary_classification_scores(y_true, y_pred, y_prob=None):
    """
    Print the Accuracy, F1, Precision and Recall
    scores for true labels `y_true` (array-like of ints)
    and predictions `y_pred` (array-like of ints).
    """
    
    scores = [accuracy_score, f1_score, precision_score, recall_score]
    for scorer in scores:
        print('{}: {:.5f}'.format(scorer.__name__, scorer(y_true, y_pred)))
    
    if type(y_prob) != type(None):
        print('{}: {:.5f}'.format(roc_auc_score.__name__, roc_auc_score(y_true, y_prob)))

        
def processs_leggo_data(folder_path, output_prefix=None, return_df=True, verbose=True):
    """
    Load and process data about proposições and tramitações
    produced by leggoR. Return it or save it to files.
    
    Input
    -----
    
    folder_path : str
        Path to directory containing the leggoR CSV 
        files 'proposicoes.csv' and 'trams.csv'.
    
    output_prefix : str or None
        If not None, write the processed data to 
        files with the given prefix (which should include 
        the path, e.g. '../dados/processed/congresso_trams_').
        The data is split by type of proposição (e.g. MPV, PL, 
        etc.).
    
    return_df : bool
        Whether or not to return a DataFrame containing all
        the processed leggoR data (i.e. the tramitação of 
        proposições).
        
    verbose : bool
        Whether to print a log while running this function 
        or not.
    """
    
    # Carrega dados de proposições (sem as tramitações):
    if verbose:
        print('Carregando dados das proposições...')
    proposicoes = pd.read_csv(os.path.join(folder_path, 'proposicoes.csv'))

    # Create a dict to group ID leggos referring to the same proposição:
    # (sometimes when a proposição changes from camara to senado and then back to camara, they get a different id_leggo)
    if verbose:
        print('Identificando proposições com múltiplos leggo IDs...')
    leggo_id_grouper = build_leggo_id_unifier(proposicoes)

    # Create a unique link to ID leggos:
    proposicoes['unique_id_leggo'] = proposicoes['id_leggo'].map(leggo_id_grouper)
    leggo_ids = proposicoes[['id_ext', 'unique_id_leggo']].drop_duplicates().set_index('id_ext')

    # Carrega e prepara os dados de tramitações:
    if verbose:
        print('Carregando dados de tramitações...')
    trams = pd.read_csv(os.path.join(folder_path, 'trams.csv'), low_memory=False)
    trams['data'] = pd.to_datetime(trams['data'])

    # Confirma que uma id_ext corresponde à tramitação em uma casa específica:
    #assert trams.groupby('id_ext')['casa'].nunique().max() == 1

    # Existem proposições sem id do leggo:
    n_missing = len(set(trams['id_ext']) - set(proposicoes['id_ext']))
    if n_missing > 0:
        print('Existem {} proposições na base de tramitações não listadas na base de proposições.'.format(n_missing))

    # Identifica a tramitação da mesma proposição em múltiplas casas via unique ID leggo:
    if verbose:
        print('Juntando IDs leggo únicos à base de tramitações...')
    congresso_trams = trams.join(leggo_ids, on='id_ext')
    assert len(congresso_trams) == len(trams)

    # Preenche os unique ID leggo faltantes com IDs externos.
    congresso_trams['unique_id_leggo'].fillna(congresso_trams['id_ext'].astype(str), inplace=True)

    # Cria série de tipo de proposições:
    # (vamos padronizar os tipos para cada proposição. Às vezes, a mesma proposição é chamada de PL, depois de PLC, ou de MPV e depois de PLV)
    if verbose:
        print('Padronizando tipo de proposição e incluindo-o nos dados...')
    sigla_tipo_order  = {'MPV': 0, 'PEC': 1, 'PLP':2, 'PDS': 3, 'PLV': 4, 'PLN': 5, 'PL': 6, 'PLC': 7, 'PLS': 8, 'PDL': 9, 'PDC': 10, 'PFC': 11, 'PRS': 12, 'PRC': 13, 'SUG': 14, 'INC': 15}
    sigla_tipo_series = proposicoes.groupby('unique_id_leggo')['sigla_tipo'].unique().apply(lambda x: xu.pick_representative(x, sigla_tipo_order))
    # Junta à base:
    congresso_trams = congresso_trams.join(sigla_tipo_series, on='unique_id_leggo')

    # Preenche tipos faltantes com MPV:
    # (verificamos que esse é o caso para os dados que tínhamos).
    # (A função de verificação utiliza um serviço de consulta de base de dados privativo do gabinete compartilhado.)
    #check_missing_prop_types(congresso_trams, 'camara')
    #check_missing_prop_types(congresso_trams, 'senado')
    congresso_trams['sigla_tipo'].fillna('MPV', inplace=True)

    # Save the cleaned tramitações to files, separated by type of tramitação:
    if type(output_prefix) != type(None):
        if verbose:
            print('Salvando dados processados por tipo de proposição...')
        filename = output_prefix + '.csv'
        congresso_trams.to_csv(filename, index=False)
        #save_trams_by_tipo(congresso_trams, ['MPV', 'PLV'], output_prefix)
        #save_trams_by_tipo(congresso_trams, 'PEC', output_prefix)
        #save_trams_by_tipo(congresso_trams, 'PDS', output_prefix)
        #save_trams_by_tipo(congresso_trams, 'PLN', output_prefix)
        #save_trams_by_tipo(congresso_trams, 'PLP', output_prefix)
        #save_trams_by_tipo(congresso_trams, ['PL', 'PLC', 'PLS'], output_prefix)
    
    if verbose:
        print('Pronto!')
    # Return the DataFrame, if requested:
    if return_df:
        return congresso_trams

    
def filter_and_time_tramitacoes(tramitacoes, relevant_cols, event_identifier, verbose=False):
    """
    Filter the `tramitacoes` DataFrame, selecting: 
    - the `relevant_cols` (list of str) columns; 
    - the rows with no missing dates and with at least 
      one non-missing value in one of the columns 
      `event_identifier` (list of str).
    
    Also replace specific 'local' names such as 
    'MPV9548' (i.e. comissões especiais) with a 
    generic name.
    
    Then add the column of time interval (in days) 
    between subsequent events, 'delta_days'.
    """

    # Select events (ação legislativa) that have at least one entry in one of the `event_identifier` columns:
    if verbose:
        print('Selecting rows with relevant tramitações...')
    relevant_tramitacoes = select_relevant_events(tramitacoes, event_identifier)[relevant_cols]

    # Standardize comissões especiais to a single label:
    if verbose:
        print('Standardizing the name of comissões especiais...')
    relevant_tramitacoes['local']       = std_comissao_especial(relevant_tramitacoes['local'])
    relevant_tramitacoes['sigla_local'] = std_comissao_especial(relevant_tramitacoes['sigla_local'])

    # Add time interval between events:
    if verbose:
        print('Adding columns with time interval between events...')
    relevant_tramitacoes = relevant_tramitacoes.join(get_time_interval(relevant_tramitacoes))

    # Fill missing values (and remove missing dates):
    if verbose:
        print('Fill missing values with place holder and remove rows with missing dates...')
    relevant_tramitacoes = fillna_relevant_tram(relevant_tramitacoes, event_identifier)
    
    return relevant_tramitacoes


def prepare_tramitacoes_data(tramitacoes, relevant_cols, event_identifier, sorter_cols, next_days, avg_aval_timestep, max_date, max_nivel, last_n_actions, verbose=False):
    """
    Take a DataFrame `tramitacoes` of processed leggoR 
    data about tramitação de proposições (see 
    `processs_leggo_data` funtion) and transform it 
    by filtering relevant information (rows and columns) 
    and by adding assessment date and target (y) values.
    
    ATTENTION: this function multiplies number of input rows 
    by associating to each tramitação multiple assessment
    dates (i.e. the date for which we want to predict the 
    'future'). This might increase the memory usage 
    significantly.
    
    Input
    -----
    
    tramitacoes : DataFrame
        The DataFrame contaning tramitações for 
        multiple proposições, already cleaned by 
        `processs_leggo_data` function.
    
    relevant_cols : list of str
        The list of columns from `tramitacoes` to
        output. It must contain the columns specified
        below.
    
    event_identifier : list of str
        List of columns where at least one non-missing
        value must show up in order to keep that row in
        the output. In other words, these values identify 
        that something relevant happened with the proposição
        (i.e. an event).
        E.g.: ['descricao_situacao', 'evento', 'titulo_evento']
    
    sorter_cols : list of str
        List of columns to be used to chronologically sort
        the tramitações, in the order of precedence.
        E.g.: ['data', 'sequencia']
    
    next_days : int
        Number of days after the assessment date for which 
        we will look for an important event. If such an 
        event if found, the target column 'acao_futura' 
        will contain a 1. Otherwise, it will contain a 0.
    
    avg_aval_timestep : int
        Average number of days between two consecutive assessment
        dates. The actual assessment dates are randomly drawn 
        from all dates between the proposição's tramitação oldest 
        one to `max_date`.
        
    max_date : str
        Last assessment date, in format '%Y-%m-%d'. This date should 
        not exceed the date the data was collected.
    
    max_nivel : float
        The maximum value of column 'nivel' in `tramitacoes`
        (where lower values indicate more important events) for 
        which an event will be considered important for target 
        building (i.e. for setting the y value).
        
    last_n_actions : int
        For each proposição and assessment date, how many events 
        will be kept as historical data (X) before each assessment
        date. If the number of available past events is lower 
        than `last_n_actions`, pad it with standard 'empty' rows.
        
    verbose: bool
        Whether or not to print log messages when running this 
        function.
        
    Return
    ------
    
    fixed_tramitacoes : DataFrame
        DataFrame with a fixed number of past events for 
        each proposição and a target column 'acao_futura'.
        Each instance is uniquely identified by the 
        proposição AND assessment date (that is, the columns 
        'unique_id_leggo' AND 'data_aval'. The number of rows
        corresponding to a single instance is `last_n_actions`.
    """
    
    # Select relevant past information and standardize names of comissões:
    if verbose:
        print('Filtering columns and rows and standardizing locais...')
    relevant_tramitacoes = filter_and_time_tramitacoes(tramitacoes, relevant_cols, event_identifier, verbose)

    # Select all proposições from input DataFrame:
    # (this can be changed if there are too many proposições to fit in the memory)
    all_ids = relevant_tramitacoes['unique_id_leggo'].drop_duplicates()

    # Note que o código abaixo pode aumentar o tamanho do DataFrame significativamente.
    # Isso pode levar a um estouro da memória caso o número de linhas em `relevant_tramitacoes` for muito grande.
    
    # Add assessment date, the time difference between them and previous events and the target value 
    # (whether or not somethind relevant happened in the days following the assessment date).
    if verbose:
        print('Adding assessment dates, building target and standardizing past data size...')
    fixed_tramitacoes = augment_create_target_and_crop(relevant_tramitacoes, all_ids, next_days, avg_aval_timestep, max_date, max_nivel, last_n_actions, sorter_cols)

    return fixed_tramitacoes
