import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import TimeSeriesSplit

def indexing(X, indices):
    """
    Returns row of X using indices
    
    Arguments:
    X : dataframe or array to sample rows
    indices : int or array of indices
    
    Returns:
        subset of X on axis 0
    """
    
    if hasattr(X, 'iloc'):
        X = X.iloc[indices]
    else:
        X = X[indices]
    return X

def evaluate(model, X, y=None):
    """
    Fits, predicts and evaluates the results of a model on time-series cross-validation.
    
    Arguments:
    model : model to fit, predict and evaluate
    X : time-series dataframe or array to fit and predict
    y : target values
    
    Returns:
    results : tuple of std, mape, predicted values
    """
    
    tscv = TimeSeriesSplit(n_splits=5, test_size=1) 

    mape_errors = []
    predicted_values = []
    for train, test in tscv.split(X):
        
        X_train = indexing(X, train)
        X_test = indexing(X, test)
        y_train = indexing(y, train)
        y_test = indexing(y, test)
        
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        predicted_values.append(prediction[0])
        actual = y_test
    
        mape_errors.append(mape(actual, prediction))
        
    std_errors = np.std(mape_errors)
    mape_errors = np.mean(mape_errors)
    
    results = (std_errors, mape_errors, predicted_values)
    
    return results

def plot_results(x, y_true, y_pred, title):
    """
    Plots actual and predicted results of a model
    
    Arguments:
    x : x-axis
    y_true : actual target values
    y_pred : predicted target values
    title : title for a graph
    """
    
    sns.set_style('darkgrid')
    sns.set(font_scale = 1.2)
    
    plt.figure(figsize=(12,8))
    plt.plot(x, y_true, label='Actual')
    plt.plot(x, y_pred, label='Predicted')
    plt.tick_params(axis='x', rotation=70)
    plt.legend(loc='best')
    plt.title(title)
    plt.show()
    
def make_lags(ts, lags):
    """
    Makes lags of time-series data
    
    Arguments:
    ts : series to make lags of it
    lags : list of numbers of periods to shift 
    
    Returns:
    lagged_df : dataframe of lagged series
    """
    
    lagged_df = pd.concat(
        {
            f'{ts.name}_lag_{i}': ts.shift(i)
            for i in lags
        },
        axis=1)
    
    return lagged_df

def get_X_y_from_lagged_df(df_list, target_ts):
    """
    Gets X, y from dataframe with lags
    
    Arguments:
    df_list : dataframes (with lags) to concatenate 
    target_ts : target dataframe for y
    
    Returns:
    X : concatenated dataframe
    y : values from target dataframe
    """
    
    res = pd.concat(df_list, axis=1)
    res.dropna(inplace=True)
    date_to_start_from = res.iloc[0].name
    X, y = res, target_ts[target_ts.index >= date_to_start_from]
    
    return X, y


def difference(ts, periods=1):
    """
    Calculates the difference of a Series element compared with another element in the Series (default is element in previous row).
    
    Arguments:
    ts : time-series
    periods : periods to shift for calculating difference
    
    Returns:
    result_df : first differences of the time-series dataframe.
    """
    
    result_df = pd.DataFrame(ts.diff(periods=periods))
    return result_df