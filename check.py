import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import pmdarima.arima
from math import sqrt
import scipy as sc
import seaborn as sns


df=pd.read_csv('Book1.csv')
df['Month']= pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
df.set_index(df['NPL'].asfreq('1M').index,inplace=True)
train,test=train_test_split(df['NPL'],test_size=.2,shuffle=False)

#1
model1=pmdarima.arima.auto_arima(df['NPL'].astype('float64'), seasonal=True, out_of_sample_size=int(len(df['NPL'])*0.2))
model1.summary()
prediction = pd.DataFrame(model1.predict(n_periods=len(test)),index=test.index)
modelfitting=pd.DataFrame(model1.predict_in_sample(),index=df['NPL'].index)
modelfitting.columns=['fitvalue']
plt.plot(train)
plt.plot(test)
plt.plot(modelfitting['fitvalue'])
modelfitting
from sklearn.metrics import r2_score
print(r2_score(test, modelfitting['fitvalue'][-19:len(modelfitting['fitvalue'])]))
from sklearn.metrics import mean_squared_error
rmse=(sqrt(mean_squared_error(test,modelfitting['fitvalue'][-19:len(modelfitting['fitvalue'])])))
print('Test RMSE: %.3f' % rmse)
prediction

#manual
model = pmdarima.arima.ARIMA(order=(0, 0, 1), seasonal_order=(0,0,0,12), out_of_sample_size=int(len(df['NPL'])*0.2))
model.fit(df['NPL'])
model.summary()
prediction = pd.DataFrame(model.predict(n_periods=len(test)),index=test.index)
modelfitting=pd.DataFrame(model.predict_in_sample(),index=df['NPL'].index)
modelfitting.columns=['fitvalue']
plt.plot(train)
plt.plot(modelfitting['fitvalue'])
plt.plot(test)
plt.plot(modelfitting['fitvalue'][-19:(len(modelfitting['fitvalue']))])
modelfitting
from sklearn.metrics import r2_score
print(r2_score(test, modelfitting['fitvalue'][-19:(len(modelfitting['fitvalue']))]))
from sklearn.metrics import mean_squared_error
rmse=(sqrt(mean_squared_error(test,modelfitting['fitvalue'][-19:len(modelfitting['fitvalue'])])))
print('Test RMSE: %.3f' % rmse)
modelfitting


#2 mod auto arima
model1=pmdarima.arima.auto_arima(df['NPL'].astype('float64') ,start_p=0,
                                 d=0, start_q=0,max_p=5, max_d=5, max_q=5,
                                 start_P=0,D=0, start_Q=0, max_P=5, max_D=5,
                                 max_Q=5, m=12, seasonal=True,error_action='warn',
                                 trace = True,supress_warnings=True,stepwise = True)
model1.summary()
prediction = pd.DataFrame(model1.predict(n_periods=len(test)),index=test.index)
modelfitting=pd.DataFrame(model1.predict_in_sample(),index=df['NPL'].index)
modelfitting.columns=['fitvalue']
plt.plot(train)
plt.plot(test)
plt.plot(modelfitting['fitvalue'])
modelfitting
prediction
from sklearn.metrics import r2_score
print(r2_score(test, modelfitting['fitvalue'][-len(test):len(modelfitting['fitvalue'])]))
from sklearn.metrics import mean_squared_error
rmse=(sqrt(mean_squared_error(test,modelfitting['fitvalue'][-len(test):len(modelfitting['fitvalue'])])))
print('Test RMSE: %.3f' % rmse)


#het
import pylab
import scipy.stats as stats
stats.probplot(df['NPL'], dist="norm", plot=pylab)
pylab.show()

#outlier
def outlier_treatment(datacolumn):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn , [25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range,upper_range

lowerbound,upperbound=outlier_treatment(df['NPL'])
print(df['NPL'][(df['NPL'] < lowerbound) | (df['NPL'] > upperbound)])
df.drop(df['NPL'][ (df['NPL']> upperbound) | (df['NPL'] < lowerbound) ].index , inplace=True)

#boxcox
fitteddata,fittedlambda=sc.stats.boxcox(df['NPL'])

fig, ax = plt.subplots(1, 2)

sns.distplot(df['NPL'], hist=False, kde=True,
             label="Non-Normal", ax=ax[0])

sns.distplot(fitteddata, hist=False, kde=True,
             label="Normal",ax=ax[1])
fitteddata=pd.DataFrame(fitteddata)
fitteddata.set_index(df['NPL'].index,inplace=True)
fitteddata.columns=['datatransform']
plt.plot(fitteddata['datatransform'],color='green')
plt.plot(df['NPL'],color='red')

#grangar
from statsmodels.tsa.stattools import grangercausalitytests
df['monthtest'] = df.index
df['monthtest']=df['monthtest'].dt.month
grangercausalitytests(df[['NPL', 'monthtest']], maxlag=12)


# first difference
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

df['NPL 1'] = df['NPL'] - df['NPL'].shift(1)
adf_check(df['NPL 1'].dropna())
df['NPL'].plot()
plt.show()

# autocorrelation and Partial Autocorrelaction plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig_first = plot_acf(df["NPL"].dropna())
fig_second = plot_pacf(df["NPL"].dropna())
plt.show()

# Plot residual errors
residuals = pd.DataFrame(model.resid())
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

#accuract metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae,
            'mpe': mpe, 'rmse':rmse,
            'corr':corr, 'minmax':minmax})

forecast_accuracy(modelfitting['fitvalue'][-len(test):len(modelfitting['fitvalue'])], test)