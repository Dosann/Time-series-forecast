import pandas as pd
from src.model_utils import difference
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from fbprophet import Prophet

class ExpEstimator(BaseEstimator):
    """Holt Winter's Exponential Smoothing."""
    
    def __init__(
        self,
        trend=None, 
        seasonal=None, 
        seasonal_periods=None, 
        smoothing_level=None,
        smoothing_trend=None,
        smoothing_seasonal=None
    ):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.smoothing_level = smoothing_level
        self.smoothing_trend = smoothing_trend
        self.smoothing_seasonal = smoothing_seasonal

    def fit(self, X, y=None):
        """Fit the model using exponential smoothing."""
        
        self.simple_exp = ExponentialSmoothing(
            endog=X,
            trend=self.trend, 
            seasonal=self.seasonal, 
            seasonal_periods=self.seasonal_periods
        ).fit(
            smoothing_level=self.smoothing_level, 
            smoothing_trend=self.smoothing_trend, 
            smoothing_seasonal=self.smoothing_seasonal
        )
        return self

    def predict(self, X):
        """
        Predict using exponential smoothing.
        
        Arguments:
        X : array_like
            Samples.
       
        Returns:
        predicted_values : array_like
        """
        
        return self.simple_exp.forecast(len(X))
    
    
class TargetTransform(BaseEstimator, TransformerMixin):
    """Target Transformer."""
    
    def fit(self, X, y=None):
        """Remembers the last trained target value."""
        
        self.last_trained = X[-1]
        return self

    def transform(self, X):
        """Calculates the difference of a Series element compared with previous element in the Series."""
        
        diff = difference(pd.DataFrame(X)).fillna(0)
        return diff
    
    def inverse_transform(self, X):
        """Added to predicted value the last saved trained target value"""

        return (pd.DataFrame(X) + self.last_trained)
    
    
class ProphetEstimator(BaseEstimator):    
    """FbProphet."""
        
    def __init__(
        self,
        holidays=None
    ):
        self.holidays = holidays
        
    def fit(self, X, y=None):   
        """Fit the model on dataframe with columns 'ds' and 'y' using FbProphet."""
        
        df_train = pd.DataFrame(X).rename(columns = {0: 'ds', 1: 'y'})
        self.prophet = Prophet(holidays=self.holidays).fit(df=df_train)
        return self
    
    def predict(self, X):
        """
        Predict on future date (the beggining of the next month) using FbProphet.
        
        Returns:
        predicted_values : array_like
        """
        
        future_date = self.prophet.make_future_dataframe(periods=1, freq='MS', include_history=False)
        predictions = self.prophet.predict(future_date)
        return predictions.yhat.values