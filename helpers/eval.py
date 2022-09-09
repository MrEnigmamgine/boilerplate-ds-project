import pandas as pd
import numpy as np

#####################################################
#                  MODEL EVALUATION                 #
#####################################################

class BaselineModel:
    """ A simple class meant to mimic sklearn's modeling methods so that I can standardize my workflow.
    Assumes that you are fitting a single predictor.  
    For multiple predictors you will need multiple instances of this class.
    
    TODO: Handle multi-dimensional predictors
    TODO: Handle saving feature names
    """
    def __init__(self, method='mean'):
        """Initializes the model with the aggregation function defined, which will be used for fitting later."""
        self.method = method

    def fit(self, x, y):
        """Calculates the baseline for the target variable and assigns it to this instance."""
        if len(y.shape) == 1:
            self.baseline = y.agg(func=self.method)[0]
            self.baseline_proba = (y == self.baseline).mean()
        else:
             raise ValueError('Expected a 1 dimensional array.')

    def predict(self, x):
        """Always predicts the baseline value."""
        n_predictions = len(x)
        return np.full((n_predictions), self.baseline)

    def predict_proba(self, x, invert=False):
        """For classification problems, a probability prediction."""
        n_predictions = len(x)

        if not invert:
            return np.full((n_predictions), self.baseline_proba)
        else:
            return np.full((n_predictions), 1- self.baseline_proba)

def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """Standardises the evaluation of a model's metrics."""
    from sklearn import metrics
    y = actual
    yhat = predicted
    resid_p = y - yhat
    sum_of_squared_errors = (resid_p**2).sum()
    error_metrics = {
        'max_error': metrics.max_error(actual, predicted),
        'sum_squared_error' : sum_of_squared_errors,
        'mean_squared_error' : metrics.mean_squared_error(actual, predicted),
        'root_mean_squared_error' : metrics.mean_squared_error(actual, predicted, squared=False),
        'mean_aboslute_error' : metrics.mean_absolute_error(actual, predicted),
        'r2_score' : metrics.r2_score(actual, predicted, force_finite=False)
    }

    return error_metrics