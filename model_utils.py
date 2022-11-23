import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import TimeSeriesSplit

def evaluate(model, X, y=None):
    tscv = TimeSeriesSplit(n_splits=5, test_size=1) 

    mape_errors = []
    predicted_values = []
    for train, test in tscv.split(X):
        model.fit(X[train], y[train])
        prediction = model.predict(X[test])
        predicted_values.append(prediction[0])
        
        actual = y[test]
    
        mape_errors.append(mape(actual, prediction))
        
    std_errors = np.std(mape_errors)
    mape_errors = np.mean(mape_errors)
    
    return (std_errors, mape_errors, predicted_values)

def plot_results(x, y_true, y_pred, title):
    
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
    return pd.concat(
        {
            f'{ts.name}_lag_{i}': ts.shift(i)
            for i in lags
        },
        axis=1)


