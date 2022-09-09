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
            
    # cat_cols.sort()
    # num_cols.sort()
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


def get_upper_outliers(s, k=1.5, bound=False):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    if bound:
        return upper_bound
    return s.apply(lambda x: max([x - upper_bound, 0]))

def get_lower_outliers(s, k=1.5, bound=False):
    '''
    Given a series and a cutoff value, k, returns the lower outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the lower bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    if bound:
        return lower_bound
    return s.apply(lambda x: min([x - lower_bound, 0]))


def build_upper_outliers(df, k=1.5):
    """For the columns that are numerical return a dataframe that contains only the upper outliers defined by k*IQR.
    """
    out = pd.DataFrame()
    for col in df.select_dtypes('number'):
        colname_upper = col + '_upper_outliers'
        out = pd.concat([out, get_upper_outliers(df[col], k)], axis=1)
        out = out.rename(columns={col: colname_upper})
    return out


def build_lower_outliers(df, k=1.5):
    """For the columns that are numerical return a dataframe that contains only the lower outliers defined by k*IQR.
    """
    out = pd.DataFrame()
    for col in df.select_dtypes('number'):
        colname_lower = col + '_lower_outliers'
        out = pd.concat([out, get_lower_outliers(df[col], k)], axis=1)
        out = out.rename(columns={col: colname_lower})
    return out

def build_all_outliers(df, k=1.5):
    d1 = build_upper_outliers(df, k)
    d2 = build_lower_outliers(df, k)
    out = pd.concat((d1,d2), axis=1)
    return out


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

def train_scaler(df, kind='min_max'):
    """Quickly build a scaler without worrying about importing the right thing.
    Will fit to the entire dataframe so you should only pass the columns you wish to scale."""
    match kind:
        case 'min_max':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        case 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
    scaler.fit(df)
    return scaler

def scale_df(df, scaler):
    """Same as scaler.transform(), but returns a dataframe with index and columns preserved."""
    X = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns )
    return X

def drop_upper_outliers(df, cols, k=1.5):
    #function to detect and handle oulier using IQR rule
    for col in df[cols]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        df = df[(df[col] < upper_bound)]
    return df

def drop_lower_outliers(df, cols, k=1.5):
    #function to detect and handle oulier using IQR rule
    for col in df[cols]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound =  q1 - k * iqr     
        df = df[(df[col] > lower_bound)]
    return df

def drop_upper_and_lower_outliers(df, cols, k=1.5):
    #function to detect and handle oulier using IQR rule
    for col in df[cols]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        lower_bound =  q1 - k * iqr     
        df = df[(df[col] < upper_bound) & (df[col] > lower_bound)]
    return df

#####################################################
#                  DATA EXPLORATION                 #
#####################################################

def scatter_vs_target(df, target, cat=None):
    """Plots a target variable against ever other variable.
    Blanket exploration function."""
    for col in df:
        sns.scatterplot(data=df, x=col, y=target, hue=cat)
        plt.show()

def anova_variance_in_target_for_cat(df, target, cat, alpha=0.05):
    """Quickly test a target against the categories in another column."""
    from scipy.stats import f_oneway
    s= df[cat]
    vals = s.sort_values().unique()
    subsets = [df[s == vals[x]][target] for x, v in enumerate(vals)]
    stat, p = f_oneway(*subsets)
    result={'reject': p < alpha,
            'h0' : f"There is no variance in {target} between subsets of {cat}",
            'stat_name': 'F',
            'stat': stat,
            'p_value': p,
            'alpha': alpha
        }
    return result

def ttest_target_for_each_cat(df, target, cat, alpha=0.05):
    """Quickly test each category subset against the overall mean to identify which categories are signficant."""
    from scipy.stats import ttest_1samp
    s= df[cat]
    vals = s.sort_values().unique()
    subsets = [df[s == vals[x]][target] for x, v in enumerate(vals)]
    mean = df[target].mean()
    out = {}
    for i, subset in enumerate(subsets):
        stat, p = ttest_1samp(subset, mean)
        result={'reject': p < alpha,
                'h0' : f"The mean of {target} for {cat}:{vals[i]} is the same as the overall population",
                'stat_name': 'F',
                'stat': stat,
                'p_value': p,
                'alpha': alpha
            }
        out[str(vals[i])] = result
    return out

def chi2_test(s1, s2, alpha=0.05):
    """Quickly determine if two samples are dependent of one another."""
    from scipy.stats import chi2_contingency
    table = pd.crosstab(s1, s2)
    stat, p, dof, expected = chi2_contingency(table)
    result={
        'reject': p < alpha,
        'h0' : f"The two samples are independent.",
        'stat_name': 'Chi2',
        'stat': stat,
        'p_value': p,
        'alpha': alpha,
        # 'misc' : {
            # 'dof' : dof,
            # 'expected': expected,
            # 'observed': table
            # }
        }
    return result

def spearman_correllation_test(df, x, y, alpha=0.05):
    from scipy.stats import spearmanr

    stat, p = spearmanr(df[x], df[y])
    result={'reject': p < alpha,
        'h0' : f"The samples of '{x}' and '{y}' are independant",
        'stat_name': 'correlation',
        'stat': stat,
        'p_value': p,
        'alpha': alpha
    }
    return result

def shapiro_gausian_test(s, alpha=0.05):
    from scipy.stats import shapiro

    stat, p = stat, p = shapiro(s)
    result={'reject': p < alpha,
        'h0' : f"The distribution is gaussian",
        'stat_name': 'statistic',
        'stat': stat,
        'p_value': p,
        'alpha': alpha
    }
    return result

def all_the_stats(df, override_categorical=[], override_numerical=[]):
    # Initialize the dictionary that will be iteratively built
    out = {}
    # Separate columns into categorical and numerical
    coltype = get_column_types(df, override_categorical=override_categorical)
    
    # Loop through each categorical column
    for col in coltype['cat']:
        if len(df[col].value_counts()) > 1:
            cold = out[col] = {}
            
            # Run a chi2 test on every other categorical column
            this = cold['chi2'] = {}
            for target in coltype['cat']:
                if len(df[target].value_counts()) > 1:
                    if target != col:
                        this[target] = chi2_test(df[col], df[target])
            # Run an anova test on every numerical column
            this = cold['anova'] = {}
            for target in coltype['num']:
                if target != col:
                    anova = this[target] = anova_variance_in_target_for_cat(df, target, col)
                    # If we reject the null run a ttest to determine which categories are significant
                    if anova['reject'] == True:
                        anova['ttest'] = ttest_target_for_each_cat(df, target, col)

    # The loop through each numerical column
    for col in coltype['num']:
        
        cold = out[col] = {}
        
        # Repeat the Anova tests on each categorical column for readability
        this = cold['anova'] = {}
        for target in coltype['cat']:
            if len(df[target].value_counts()) > 1:
                if target != col:
                    anova = this[target] = anova_variance_in_target_for_cat(df, col, target)
                    if anova['reject'] == True:
                        anova['ttest'] = ttest_target_for_each_cat(df, col, target)

        # Run a correlation test for every other numerical column
        this = cold['spearmanr'] = {}
        for target in coltype['num']:
            if target != col:
                this[target] = spearman_correllation_test(df, target, col)

    return out

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

def build_kmeans_clusterer(df, cols, k):
    from sklearn.cluster import KMeans
    clusterer = KMeans(n_clusters=k)
    clusterer.fit(df[cols])
    return clusterer

def get_kmeans_clusters(df, cols, k=5, clusterer=None):
    if clusterer == None:
        from sklearn.cluster import KMeans
        clusterer = KMeans(n_clusters=k)
        clusterer.fit(df[cols])
    s = clusterer.predict(df[cols])
    return s


def find_k(df, cluster_vars, k_range, seed=SEED):
    from sklearn.cluster import KMeans
    sse = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=seed)

        # X[0] is our df dataframe..the first dataframe in the list of dataframes stored in X. 
        kmeans.fit(df[cluster_vars])

        # inertia: Sum of squared distances of samples to their closest cluster center.
        sse.append(kmeans.inertia_) 

    # compute the difference from one k to the next
    delta = [round(sse[i] - sse[i+1],0) for i in range(len(sse)-1)]

    # compute the percent difference from one k to the next
    pct_delta = [round(((sse[i] - sse[i+1])/sse[i])*100, 1) for i in range(len(sse)-1)]

    # create a dataframe with all of our metrics to compare them across values of k: SSE, delta, pct_delta
    k_comparisons_df = pd.DataFrame(dict(k=k_range[0:-1], 
                             sse=sse[0:-1], 
                             delta=delta, 
                             pct_delta=pct_delta))

    # plot k with inertia
    plt.plot(k_comparisons_df.k, k_comparisons_df.sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('The Elbow Method to find the optimal k\nFor which k values do we see large decreases in SSE?')
    plt.show()

    # plot k with pct_delta
    plt.plot(k_comparisons_df.k, k_comparisons_df.pct_delta, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Percent Change')
    plt.title('For which k values are we seeing increased changes (%) in SSE?')
    plt.show()

    # plot k with delta
    plt.plot(k_comparisons_df.k, k_comparisons_df.delta, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Absolute Change in SSE')
    plt.title('For which k values are we seeing increased changes (absolute) in SSE?')
    plt.show()

    return k_comparisons_df






def plot_residuals(actual, predicted):
    """Plots the residuals of a model's predictions."""
    yhat = predicted
    resid_p = actual - yhat

    fig, ax1 = plt.subplots(1, 1, constrained_layout=True, sharey=True, figsize=(7,4))
    ax1.set_title('Predicted Residuals')
    ax1.set_ylabel('Error')
    ax1.set_xlabel('Predicted Value')
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.scatter(x=yhat, y=resid_p)
    plt.show()