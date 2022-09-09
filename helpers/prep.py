import pandas as pd

SEED = 8

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

def upsample_target(df, target, val):
    """Assumes a binary case for now. Duplicates rows where the column value is the the named value to create a balance of classes."""
    from sklearn.utils import resample
    # Upsample the dfing data to balance a class imbalance
    minority_upsample = resample( df[df[target] == val],   #DF of samples to replicate
                                replace = True,         #Implements resampling with replacement, Default=True
                                n_samples = len(df[df[target]!=val])-1, #Number of samples to produce
                                random_state= 8         #Random State seed for reproducibility
                                )
    #Then glue the upsample to the original
    return pd.concat([minority_upsample, df[df[target]!=val]])