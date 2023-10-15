import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX


sunspots = pd.read_csv('sunspots.csv', parse_dates=['Date'], index_col='Date')
sunspots.index.freq = 'MS'  

train_size = int(0.8 * len(sunspots))
train, test = sunspots[:train_size], sunspots[train_size:]

order = (1, 1, 1)  # (p, d, q)
seasonal_order = (1, 1, 1, 12)  # (P, D, Q, S)

model = SARIMAX(train['Sunspots'], order=order, seasonal_order=seasonal_order)
results = model.fit()

start = len(train)
end = len(train) + len(test) - 1
predictions = results.predict(start=start, end=end, dynamic=False, typ='levels')

plt.figure(figsize=(12, 6))
plt.plot(train['Sunspots'], label='Train')
plt.plot(test['Sunspots'], label='Test')
plt.plot(predictions, label='SARIMA Predictions')
plt.legend()
plt.title('Sunspot Activity Forecast')
plt.show()
