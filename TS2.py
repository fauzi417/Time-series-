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

#Lanjut2
#list kosong
order_aic_bic = []

#loop ar
for p in range(3):
    # Loop ma
    for q in range(3):
        try:
            model = SARIMAX(df['NTF'], order=(p, 0, q))
            results = model.fit()

            # masukin aic bic ke list
            order_aic_bic.append((p, q, results.aic, results.bic))
        except:
            order_aic_bic.append((p, q, None, None))

#nama tabel
order_df = pd.DataFrame(order_aic_bic,
                        columns=['p', 'q', 'AIC', 'BIC'])

#print sort aic dan bic
print(order_df.sort_values('AIC'))
print(order_df.sort_values('BIC'))

#Metrics (selalu liat residual)
#model diagnostics
model = SARIMAX(df['NTF'], order=(1, 0, 1))
results = model.fit()
mae = np.mean(np.abs(results.resid))
me = np.mean(results.resid)             # ME
mae = np.mean(np.abs(results.resid))    # MAE
mse = np.mean((results.resid)**2)       # MSE
rmse = np.mean((results.resid)**2)**.5  # RMSE
print(mae,me,mae,mse,rmse)
print(results.summary())
#Prob Q residual tidak korelasi, Prob JB tapi residual tidak menyebar normal

results.plot_diagnostics()
plt.show()
#1. tidak ada pattern dan atas bawah lebarnya sama , OKE
#2. Histogram normal tapi kurang mendekati normal 0,1 , Tidak OKE
#3. QQ Plot gak lurus sama garis, Tidak OKE
#4. Correlogram tidak ada yg keluar, OKE


#Langkah Box Jenkins
#1. Indentification: plot data asli, grangar causality, cari lag potensial, dickey fuller, cari T paling kecil dan Pvalue <0.05
# lanjut cari ACF dan PACF potensial Order ARMA
#2.Estimation: Trial and Error ARMA fit dan cari bic aic terkecil
#3. Model Diagnostic: plot diagnostic, summary liat prob Q dan JB
#4. Decision: Model diagnostic masih ada yg belum oke ? ulang ke identification cari transformasi lain
# kalo udah oke semua lanjut 5
#5. Forecast


#SEASONAL ARIMA
df=pd.read_excel('C:/Users/PREDATOR/OneDrive/Dokumen/ts/NTF series (1).xlsx')
df['Date']= pd.to_datetime(df['Date'])
df2=df.groupby([df['Date'].dt.year, df['Date'].dt.month], as_index=False).last()
df.set_index('Date', inplace=True)
df2.set_index('Date', inplace=True)
s2 = df.groupby([lambda x: x.year, lambda x: x.month]).sum()
s2.set_index(df2['NTF'].asfreq('1M').index,inplace=True)

# analyze the data, give rough picture of the data
print(s2['NTF'].describe())
s2['NTF'].plot()
plt.show()

#decomp

decomp = seasonal_decompose(s2['NTF'].asfreq('1M'))
decomp.plot()
plt.show()

#adf
adf_check(s2['NTF'])

#differencing
#first difference
s2['NTF Diff 1']=s2['NTF']-s2['NTF'].shift(1)
adf_check(s2['NTF Diff 1'].dropna())
s2['NTF Diff 1'].plot()
plt.show()

#acf pacf diff 1
plot_acf(s2['NTF Diff 1'].dropna(),lags=20)
plot_pacf(s2['NTF Diff 1'].dropna(),lags=20)

#detrending
s2['NTF'].plot()
s2['NTF'].rolling(12).mean().plot()
dfdetrend=s2['NTF']-s2['NTF'].rolling(12).mean()
dfdetrend.plot()

#acf pacf detrending
plot_acf(dfdetrend.dropna(),lags=20)
plot_pacf(dfdetrend.dropna(),lags=20)

#seasonal difference
s2['NTF Diff 1 dan 12']=s2['NTF Diff 1']-s2['NTF Diff 1'].shift(12)
adf_check(s2['NTF Diff 1 dan 12'].dropna())
s2['NTF Diff 1 dan 12'].plot()
plt.show()

#buat model ARIMA dan SARIMA
#acf pacf diff 1 order ARMA
plot_acf(s2['NTF Diff 1'].dropna(),lags=11)
plot_pacf(s2['NTF Diff 1'].dropna(),lags=11)

#acf pacf diff 1 order seasonal
lags=[6,12,18,24]
plot_acf(s2['NTF Diff 1'].dropna(),lags=lags)
plot_pacf(s2['NTF Diff 1'].dropna(),lags=lags)

#acf pacf diff 1 + diff 12 order
plot_acf(s2['NTF Diff 1 dan 12'].dropna(),lags=11)
plot_pacf(s2['NTF Diff 1 dan 12'].dropna(),lags=11)

#acf pacf diff 1 order seasonal
lags=[12,24]
plot_acf(s2['NTF Diff 1 dan 12'].dropna(),lags=lags)
plot_pacf(s2['NTF Diff 1 dan 12'].dropna(),lags=lags)

#model
model1 = SARIMAX(s2['NTF'], order=(2, 1, 1), seasonal_order=(2,0,2,6))
results1 = model1.fit()
print(results1.summary())

model2 = SARIMAX(s2['NTF'], order=(0, 1, 0), seasonal_order=(2,1,0,12))
results2 = model2.fit()
print(results2.summary())

#model ARIMA aja
model3 = SARIMAX(s2['NTF'], order=(2, 1, 1))
results3 = model3.fit()
print(results3.summary())


# Create SARIMA mean forecast
sarima_pred1 = results1.get_prediction(start=-10,dynamic=True)
sarima_mean1 = sarima_pred1.predicted_mean


# Create SARIMA mean forecast
sarima_pred2 = results2.get_prediction(start=-10,dynamic=True)
sarima_mean2 = sarima_pred2.predicted_mean

# Create ARIMA mean forecast
arima_pred = results3.get_prediction(start=-10,dynamic=True)
arima_mean = arima_pred.predicted_mean

# Plot mean ARIMA and SARIMA predictions and observed
plt.plot(s2['NTF'], label='observed')
plt.plot(sarima_mean1.index, sarima_mean1, label='SARIMA 1')
plt.plot(sarima_mean2.index, sarima_mean2, label='SARIMA 2')
plt.plot(arima_mean.index, arima_mean, label='ARIMA')
plt.legend()
plt.show()


#GARCH
#GARCH
from arch import arch_model
import matplotlib.pyplot as plt

#model
model1 = SARIMAX(s2['NTF'], order=(2, 1, 1), seasonal_order=(2,0,2,6))
results1 = model1.fit()
print(results1.summary())

# Specify GARCH model assumptions
basic_gm = arch_model(results1.resid, p = 2, q = 1,
                      mean = 'constant', vol = 'GARCH', dist = 'normal')
# Fit the model
gm_result = basic_gm.fit(update_freq = 1)

# Display model fitting summary
print(gm_result.summary())

# Plot fitted results
gm_result.plot()
plt.show()

# Make 5-period ahead forecast
gm_forecast = gm_result.forecast(horizon = 5)

# Print the forecast variance
print(gm_forecast.variance[-1:])
