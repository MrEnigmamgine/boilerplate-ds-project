import pandas as pd


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