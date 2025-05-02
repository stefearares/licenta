import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


years = [2000, 2001, 2002, 2005, 2006, 2007, 2008, 2009]
pixels = [1200, 1325, 1500, 1600, 1580, 1610, 1625, 1610]

# Create a Series
ts = pd.Series(data=pixels, index=pd.DatetimeIndex([f"{y}-01-01" for y in years]))

# Fit ARIMA(1,1,1) as example
model = ARIMA(ts, order=(1,1,1))
res = model.fit()

# Forecast next 5 years
forecast = res.get_forecast(steps=5)
print(forecast.predicted_mean)

#how wide the uncertainty is in your prediction, if intervals too wide model not very confident
print(forecast.conf_int())

'''
Look for:

AIC/BIC: Lower is better.

p-values for AR/MA terms: < 0.05 = statistically significant.

Standard errors: Large = possibly unstable estimates.
'''
print(res.summary())
