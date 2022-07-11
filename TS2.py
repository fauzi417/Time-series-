import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_excel('C:/Users/PREDATOR/OneDrive/Dokumen/ts/NTF series (1).xlsx')
df['Date']= pd.to_datetime(df['Date'])
df=df.groupby([df['Date'].dt.year, df['Date'].dt.month], as_index=False).last()
df.set_index('Date', inplace=True)
df.set_index(df['NTF'].asfreq('1M').index,inplace=True)

# analyze the data, give rough picture of the data
print(df['NTF'].describe())
df['NTF'].plot()
plt.show()

#decomp
from statsmodels.tsa.seasonal import seasonal_decompose

decomp = seasonal_decompose(df['NTF'].asfreq('1M'))
decomp.plot()
plt.show()

# Visualize 4 Month Rolling mean adn std
time_series = df['NTF']
print(type(time_series))
time_series.rolling(12).mean().plot(label='12 month rolling mean')
time_series.rolling(12).std().plot(label='12 month rolling std')
time_series.plot()
plt.legend()
plt.show()

#outlier
def outlier_treatment(datacolumn):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn , [25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range,upper_range

lowerbound,upperbound=outlier_treatment(df['NTF'])
print(df['NTF'][(df['NTF'] < lowerbound) | (df['NTF'] > upperbound)])

# Augmented dickey fuller test
from statsmodels.tsa.stattools import adfuller


def adf_check(time_series):
    """
    Pass in a time series, returns ADF report
    """
    result = adfuller(time_series)
    print('Augmented Dickey-Fuller Test:')
    labels = ['ADF Test Statistic',
              'p-value',
              '#Lags Used',
              'Number of Observations Used']

    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))

    if result[1] <= 0.05:
        print(
            "strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
    print()


adf_check(time_series)


#heteroscedasticity
import pylab
import scipy.stats as stats
stats.probplot(df['NTF'], dist="norm", plot=pylab)
pylab.show()

#grangar
from statsmodels.tsa.stattools import grangercausalitytests
df['Date'] = df.index
df['Date']=df['Date'].dt.month
grangercausalitytests(df[['NTF', 'Date']], maxlag=12)

#differencing
#first difference
df['NTF Diff 1']=df['NTF']-df['NTF'].shift(1)
adf_check(df['NTF Diff 1'].dropna())
df['NTF Diff 1'].plot()
plt.show()
stats.probplot(df['NTF Diff 1'][1:], dist="norm", plot=pylab)
pylab.show()

#second difference
df['NTF Diff 2']=df['NTF']-df['NTF'].shift(2)
adf_check(df['NTF Diff 2'].dropna())
df['NTF Diff 2'].plot()
plt.show()
stats.probplot(df['NTF Diff 2'][2:], dist="norm", plot=pylab)
pylab.show()

#seasonal difference
df['NTF Diff seasonal']=df['NTF']-df['NTF'].shift(12)
adf_check(df['NTF Diff seasonal'].dropna())
df['NTF Diff seasonal'].plot()
plt.show()
stats.probplot(df['NTF Diff seasonal'][12:], dist="norm", plot=pylab)
pylab.show()



#autocorrelation and Partial Autocorrelaction plot

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
fig_first = plot_acf(df['NTF'].dropna())
fig_second = plot_pacf(df['NTF'].dropna())
plt.show()


#Lanjutan
#first Differencing
#autocorrelation and Partial Autocorrelaction plot

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
fig_first = plot_acf(df['NTF Diff 1'].dropna())
fig_second = plot_pacf(df['NTF Diff 1'].dropna())
plt.show()

#fit dan summary
from statsmodels.tsa.statespace.sarimax import SARIMAX
mod = SARIMAX(df['NTF Diff 1'], order=(1, 0,1))
res = mod.fit()
print(res.summary())

#Exog
modexog = SARIMAX(df['NTF Diff 1'], order=(1, 0,1),exog=df['Acct'])
resexog = modexog.fit()
print(resexog.summary())

#coba forecast data terakhir
#one step forecast
one_step_forecast=res.get_prediction(start=-10)
mean_forecast = one_step_forecast.predicted_mean
confidence_intervals = one_step_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower NTF Diff 1']
upper_limits = confidence_intervals.loc[:,'upper NTF Diff 1']
print(mean_forecast)

#plot forecast
plt.plot(df['NTF Diff 1'].index,df['NTF Diff 1'],label='observed')
plt.plot(mean_forecast.index,mean_forecast,color='r',label='forecast')
plt.fill_between(mean_forecast.index,lower_limits,upper_limits,color='pink')
plt.xlabel('Date')
plt.ylabel('NTF diff 1')
plt.legend()
plt.show()


#dynamic forecast
dynamic_forecast = res.get_prediction(start=-10, dynamic=True)
mean_forecast = dynamic_forecast.predicted_mean
confidence_intervals = dynamic_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower NTF Diff 1']
upper_limits = confidence_intervals.loc[:,'upper NTF Diff 1']
print(mean_forecast)

#plot forecast
plt.plot(df['NTF Diff 1'].index,df['NTF Diff 1'],label='observed')
plt.plot(mean_forecast.index,mean_forecast,color='r',label='forecast')
plt.fill_between(mean_forecast.index,lower_limits,upper_limits,color='pink')
plt.xlabel('Date')
plt.ylabel('NTF diff 1')
plt.legend()
plt.show()

#forecast kedepan (pasti dynamic)
dynamic_forecast_forward=res.get_forecast(steps=10).predicted_mean
plt.plot(df['NTF Diff 1'].index,df['NTF Diff 1'],label='observed')
plt.plot(dynamic_forecast_forward.index,dynamic_forecast_forward,color='r',label='forecast')
plt.xlabel('Date')
plt.ylabel('NTF diff 1')
plt.legend()
plt.show()

#forecast kedepan data asli
asli_int_forecast = np.cumsum(dynamic_forecast_forward)
asli_value_forecast = asli_int_forecast + df['NTF'].iloc[-1]
plt.plot(df['NTF'].index,df['NTF'],label='observed')
plt.plot(asli_value_forecast.index,asli_value_forecast,color='r',label='forecast')
plt.xlabel('Date')
plt.ylabel('NTF')
plt.legend()
plt.show()

#ARIMA, langsung forecast differencing nya
from statsmodels.tsa.statespace.sarimax import SARIMAX
mod = SARIMAX(df['NTF'], order=(1, 1, 1))
res = mod.fit()
print(res.summary())


#SEASONAL
#autocorrelation and Partial Autocorrelaction plot

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
fig_first = plot_acf(df['NTF Diff seasonal'].dropna())
fig_second = plot_pacf(df['NTF Diff seasonal'].dropna())
plt.show()

#fit dan summary
from statsmodels.tsa.arima.model import ARIMA
mod = ARIMA(df['NTF Diff seasonal'], order=(1, 0,2))
res = mod.fit()
print(res.summary())

#Exog
modexog = ARIMA(df['NTF Diff seasonal'], order=(1, 0,2),exog=df['Acct'])
resexog = modexog.fit()
print(resexog.summary())

#coba forecast data terakhir
#one step forecast
one_step_forecast=res.get_prediction(start=-10)
mean_forecast = one_step_forecast.predicted_mean
confidence_intervals = one_step_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower NTF Diff seasonal']
upper_limits = confidence_intervals.loc[:,'upper NTF Diff seasonal']
print(mean_forecast)

#plot forecast
plt.plot(df['NTF Diff seasonal'].index,df['NTF Diff seasonal'],label='observed')
plt.plot(mean_forecast.index,mean_forecast,color='r',label='forecast')
plt.fill_between(mean_forecast.index,lower_limits,upper_limits,color='pink')
plt.xlabel('Date')
plt.ylabel('NTF diff seasonal')
plt.legend()
plt.show()


#dynamic forecast
dynamic_forecast = res.get_prediction(start=-10, dynamic=True)
mean_forecast = dynamic_forecast.predicted_mean
confidence_intervals = dynamic_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower NTF Diff seasonal']
upper_limits = confidence_intervals.loc[:,'upper NTF Diff seasonal']
print(mean_forecast)

#plot forecast
plt.plot(df['NTF Diff seasonal'].index,df['NTF Diff seasonal'],label='observed')
plt.plot(mean_forecast.index,mean_forecast,color='r',label='forecast')
plt.fill_between(mean_forecast.index,lower_limits,upper_limits,color='pink')
plt.xlabel('Date')
plt.ylabel('NTF diff seasonal')
plt.legend()
plt.show()

#forecast kedepan (pasti dynamic)
dynamic_forecast_forward=res.get_forecast(steps=10).predicted_mean
plt.plot(df['NTF Diff seasonal'].index,df['NTF Diff seasonal'],label='observed')
plt.plot(dynamic_forecast_forward.index,dynamic_forecast_forward,color='r',label='forecast')
plt.xlabel('Date')
plt.ylabel('NTF diff seasonal')
plt.legend()
plt.show()


#forecast kedepan data asli
asli_int_forecast = np.cumsum(dynamic_forecast_forward)
asli_value_forecast = asli_int_forecast + df['NTF'].iloc[-1]
plt.plot(df['NTF'].index,df['NTF'],label='observed')
plt.plot(asli_value_forecast.index,asli_value_forecast,color='r',label='forecast')
plt.xlabel('Date')
plt.ylabel('NTF')
plt.legend()
plt.show()

#ARIMA, langsung forecast differencing nya
from statsmodels.tsa.statespace.sarimax import SARIMAX
mod = SARIMAX(df['NTF Diff seasonal'], order=(1, 0,2))
res = mod.fit()
print(res.summary())

from statsmodels.tsa.statespace.sarimax import SARIMAX
mod = SARIMAX(df['NTF'], order=(1, 12, 2))
res = mod.fit()
print(res.summary())

#, order=(1, 0, 2), seasonal_order=(1,1,0,12)
