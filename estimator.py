from sklearn.base import BaseEstimator
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class ExpEstimator(BaseEstimator):
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
        self.simple_exp = ExponentialSmoothing(
            endog=X,
            trend=self.trend, 
            seasonal=self.seasonal, 
            seasonal_periods=self.seasonal_periods,
            
        ).fit(
            smoothing_level=self.smoothing_level, 
            smoothing_trend=self.smoothing_trend, 
            smoothing_seasonal=self.smoothing_seasonal
        )
        return self

    def predict(self, X):
        return self.simple_exp.forecast(len(X))