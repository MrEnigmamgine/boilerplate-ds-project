import pandas as pd
import os

CSV='./data.csv'
DB= ''
SQLQUERY ="""

"""


def get_db_url(database):
    """Formats a SQL url by using the env.py file to store credentials."""
    from env import host, user, password
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    return url

def new_data():
    """Downloads a copy of data from a SQL Server.
    Relies on an env.py file and the configuration of the DB and SQLQUERY variables."""
    url = get_db_url(DB)
    df = pd.read_sql(SQLQUERY, url)
    return df

def get_data(refresh=False):
    """Returns an uncleaned copy of the data from the CSV file defined in config.
    If the file does not exist, grabs a new copy and creates the file.
    Assumes the use of a SQL query.
    """
    filename = CSV
    # if file is available locally, read it
    if os.path.isfile(filename) and not refresh:
        return pd.read_csv(filename)
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df = new_data()
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)
        # Return the dataframe to the calling code
        return df  

