# Core modules
import os
import math

# DS Modules
import numpy as np
import pandas as pd
from pydataset import data

# Visualization modules
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import datetime

# SKLearn
import sklearn as sk
from sklearn.model_selection import train_test_split

# Pandas Options
# pd.options.display.max_columns = None
# pd.options.display.max_rows = 70
# pd.options.display.float_format = '{:20,.2f}'.format

#####################################################
#               CONFIG VARIABLES                    #
#####################################################

SEED = 8
FEATURES = []
TARGETS = []
ENVFILE = './env.py'
CSV='./data.csv'
DB= ''
SQLQUERY ="""

"""
#####################################################
#               END CONFIG VARIABLES                #
#####################################################

#####################################################
#                   DATA GETTING                    #
#####################################################



# Easily load a google sheet (first tab only)
def read_google(url):
    """Copy paste the browser URL into this function and it will automagically put the first tab into pandas."""
    csv_export_url = url.replace('/edit#gid=', '/export?format=csv&gid=')
    return pd.read_csv(csv_export_url)


#####################################################
#                   DATA PREPPING                   #
#####################################################

def fix_dtypes(df):
    """convert_dtypes() chooses some slightly wonky data types that cause problems later.
    Fix the wonk by creating a new dataframe from the dataframe. """
    df = df.convert_dtypes()
    fix = pd.DataFrame(df.to_dict()) 
    return fix

def dropna_df(df):
    """Returns a dataframe free of null values where the columns have the proper dtypes"""
    df = df.dropna()
    df = fix_dtypes(df)
    return df

def handle_missing_values(df, drop_cols_threshold=0.75, drop_rows_threshold=0.75):
    """Given a dataframe and some thresholds, trims the dataframe so that any column or row with too many null values is dropped."""
    threshold = int(round(drop_cols_threshold * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold) # axis 1, or ‘columns’ : Drop columns which contain missing value
    threshold = int(round(drop_rows_threshold * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold) # axis 0, or ‘index’ : Drop rows which contain missing values.
    return df

def get_highcounts(df):
    """Returns a dataframe containing the 4 highest value counts for each column.
    Or in the case of continuous variables, the counts of 4 bins."""
    categorical_types =['object','string','bool','category'] # The dtypes we will treat as categorical. Might not be a complete list.
    d = {} # The dictionary we will build
    # Loop through each column
    for col in df:
        # and get the highest 4 (value, count) tuples using .head() and .iteritems()
        if df[col].dtype in categorical_types:
            d[col] = (  list(df[col].value_counts(dropna = False).head(4).iteritems()) )
        # Make sure there are more than 4 values before we try binning
        elif df[col].nunique() > 4:
            d[col] = (  list(df[col].value_counts(bins = 4, dropna=False).iteritems()) )
        # And then get the rest.
        else:
            d[col] = (  list(df[col].value_counts(dropna = False).head(4).iteritems()) )

    # Build the dataframe using from_dict and orient="index"
    outdf = pd.DataFrame.from_dict(d, orient='index')
    # Rename the columns for ease of access
    outdf.columns = ['highcount_'+str(col) for col in outdf]
    return outdf

def col_summary(df):
    """Returns a dataframe full of statistics about each column in a dataframe.
    Useful for scrubbing datatypes and handling nulls."""
    # Build the datatype column
    dt = pd.DataFrame(df.dtypes)
    dt.columns = ['dtype']
    # Count of nulls column
    ns = pd.DataFrame(df.isna().sum())
    ns.columns = ['null_sum']
    # Percentage of nulls column
    nm = pd.DataFrame(df.isna().mean())
    nm.columns = ['null_mean']
    # Count of unique values
    nu = pd.DataFrame(df.nunique())
    nu.columns = ['n_unique']
    # Count of possible hidden nulls
    d = {}
    for col in df:
        d[col] = df[col].apply(lambda x: str(x).strip() == '' or str(x).strip().lower() in ['na','n/a']).sum()
    hnulls = pd.DataFrame.from_dict(d, orient='index')
    hnulls.columns = ['hidden_nulls']
    # Put everything together in a dataframe
    out = pd.concat([dt, ns, nm, hnulls, nu], axis=1)
    # One more statistic that's easier to calculate after putting everything together.
    out['duplicates'] = len(df) - out['n_unique']

    return out

def nulls_by_row(df):
    """Classroom dictated function. Yet to find a use-case for it."""
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})
    rows_missing = df.merge(rows_missing,
                        left_index=True,
                        right_index=True)[['num_cols_missing', 'percent_cols_missing']]
    return rows_missing.sort_values(by='num_cols_missing', ascending=False)

def get_gotchas(df, cat_threshold=10):
    """Given a df, tries to identify potential problems in the data that might need addressed."""
    out = {
        'possible_ids': [],
        'possible_bools': [],
        'probable_categories': []
        }
    summary = col_summary(df)
    for name, row in summary.iterrows():
        if len(df) - row.loc['n_unique'] <= 1:
            out['possible_ids'].append(name)
        if row.loc['n_unique'] in [1,2]:
            out['possible_bools'].append(name)
        if row.loc['n_unique'] < cat_threshold:
            out['probable_categories'].append(name)

    return out

def get_column_types(df, override_categorical=[], override_numerical=[]):

    cat_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()
    num_cols = df.select_dtypes(exclude=['object','category','bool']).columns.tolist()

    for val in override_categorical:
        if val in num_cols:
            num_cols.remove(val)
            cat_cols.append(val)

    for val in override_numerical:
        if val in cat_cols:
            cat_cols.remove(val)
            num_cols.append(val)
            
    out = {
        'cat': cat_cols,
        'num': num_cols
    }
    return out

def all_the_hist(df):
    """Plots a histogram for every column of a DF into one figure."""
    import math
    vizcols = 5
    vizrows = math.ceil(len(df.columns)/vizcols)
    fig, ax = plt.subplots(vizrows, vizcols, figsize=(15,vizrows*4))
    for i, col in enumerate(df):
        r, c = i % vizrows, i % vizcols
        a = ax[r][c]
        a.set_title(col)
        df[col].hist(ax=a)
    fig




#####################################################
#                DATA PRE-PROCESSING                #
#####################################################

## Generic split data function
def train_test_validate_split(df, seed=SEED, stratify=None):
    """Splits data 60%/20%/20%"""
    from sklearn.model_selection import train_test_split
    # First split off our testing data.
    train, test_validate = train_test_split(
        df, 
        train_size=3/5, 
        random_state=seed, 
        stratify=( df[stratify] if stratify else None)
    )
    # Then split the remaining into train/validate data.
    test, validate = train_test_split(
        test_validate,
        train_size=1/2,
        random_state=seed,
        stratify= (test_validate[stratify] if stratify else None)
    )
    return train, test, validate





#####################################################
#                  DATA EXPLORATION                 #
#####################################################

def scatter_vs_target(df, target, cat=None):
    """Plots a target variable against ever other variable.
    Blanket exploration function."""
    for col in df:
        sns.scatterplot(data=df, x=col, y=target, hue=cat)
        plt.show()


def pop_unrejected(results):
    for column, tests in results.items():
        for test, targets in tests.items():
            to_pop = []
            for target, result in targets.items():
                if result['reject'] == False:
                    to_pop.append(target)
            for target in to_pop:
                del targets[target]
    return results

import json
class CustomJSONizer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return super().encode(bool(obj))
        
        elif isinstance(obj, np.ndarray):
            return super().encode(str(obj))
        
        elif isinstance(obj, pd.DataFrame):
            return super().encode(obj.to_dict())

        else:
            return super().default(obj)

def prettify(obj):
    return json.dumps(obj, cls=CustomJSONizer, indent=2) 


#####################################################
#                DATA MODEL PREPPING                #
#####################################################


