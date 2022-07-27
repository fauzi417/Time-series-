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

# Print the forecast variance dan mean(buat forecast pake mean)
print(gm_forecast.variance[-1:])
print(gm_forecast.mean[-1:])


#MATERI SENIN
# Obtain model estimated residuals and volatility
gm_resid = gm_result.resid
gm_std = gm_result.conditional_volatility

# Calculate the standardized residuals
gm_std_resid = gm_resid /gm_std
gm_std_resid.plot()

# Plot the histogram of the standardized residuals
plt.hist(gm_std_resid,bins=15,
         facecolor = 'orange', label = 'Standardized residuals')
plt.show()

#karena agak miring, kita ganti pake skewt
skewt_gm = arch_model(results1.resid, p = 1, q = 1,
                      mean = 'constant', vol = 'GARCH', dist = 'skewt')

skewt_result = skewt_gm.fit()
print(skewt_result.summary())

# Get model estimated volatility
skewt_vol = skewt_result.conditional_volatility

# Plot model fitting results
plt.plot(gm_std,
         color = 'red', label = 'Standardized residuals')
plt.plot(skewt_vol, color = 'gold', label = 'Skewed-t Volatility')
plt.plot(results1.resid, color = 'grey',
         label = 'Residual ARIMA', alpha = 0.4)
plt.legend(loc = 'upper right')
plt.show()

# Different mean
zero_gm = arch_model(results1.resid, p = 1, q = 1,
                      mean = 'zero', vol = 'GARCH', dist = 'skewt')
zero_result = zero_gm.fit()
print(zero_result.summary())

ar_gm = arch_model(results1.resid, p = 1, q = 1,
                      mean = 'AR',lags=1, vol = 'GARCH', dist = 'skewt')
ar_result = ar_gm.fit()
print(ar_result.summary())

ar_vol=ar_result.conditional_volatility
# Plot model fitting results
plt.plot(gm_std,
         color = 'red', label = 'Standardized residuals')
plt.plot(ar_vol, color = 'gold', label = 'AR Skewed-t Volatility')
plt.plot(results1.resid, color = 'grey',
         label = 'Residual ARIMA', alpha = 0.4)
plt.legend(loc = 'upper right')
plt.show()

#MATERI SELASA
# Asymetric shock
#GJR GARCH
skewt_gjrgm = arch_model(results1.resid, p = 1, q = 1,o=1,
                      mean = 'AR',lags=1, vol = 'GARCH', dist = 'skewt')
gjrgm_result = skewt_gjrgm.fit()
print(gjrgm_result.summary())

#EGARCH
skewt_egm = arch_model(results1.resid, p = 1, q = 1,o=1,
                      mean = 'AR',lags=1, vol = 'EGARCH', dist = 'skewt')
egm_result = skewt_egm.fit()
print(egm_result.summary())

#expanding window
forecasts={}
start_loc=15
end_loc=20
for i in range(30):
    # Specify fixed rolling window size for model fitting
    gm_result = ar_gm.fit(first_obs = start_loc,
                             last_obs = i + end_loc)
    # Conduct 1-period mean forecast and save the result
    temp_result = gm_result.forecast(horizon=1).mean
    fcast = temp_result.iloc[i + end_loc]
    forecasts[fcast.name] = fcast
# Save all forecast to a dataframe
forecast_exp = pd.DataFrame(forecasts).T

# Plot the forecast mean
plt.plot(forecast_exp, color='red')
plt.plot(results1.resid[21:51], color='green')
plt.show()

#fixed window
forecasts={}
start_loc=15
end_loc=20
for i in range(30):
    # Specify fixed rolling window size for model fitting
    gm_result = ar_gm.fit(first_obs = i + start_loc,
                             last_obs = i + end_loc)
    # Conduct 1-period variance forecast and save the result
    temp_result = gm_result.forecast(horizon=1).mean
    fcast = temp_result.iloc[i + end_loc]
    forecasts[fcast.name] = fcast
# Save all forecast to a dataframe
forecast_roll = pd.DataFrame(forecasts).T

# Plot the forecast mean
plt.plot(forecast_roll, color='red')
plt.plot(results1.resid[21:51], color='green')
plt.show()


# Plot the forecast mean expanding and rolling
plt.plot(forecast_exp, color='blue')
plt.plot(forecast_roll, color='red')
plt.plot(results1.resid[21:51], color='green')
plt.show()


######################RABU
#cek pvalue dan tvalue
print(ar_result.summary())

#eliminasi
ar_gm = arch_model(results1.resid, p = 1, q = 0,
                      mean = 'AR',lags=1, vol = 'GARCH', dist = 'skewt')
ar_result = ar_gm.fit()
print(ar_result.summary())

#validasi
#visual check
ar_result.plot()
plt.show()

#autocor
ar_resid = ar_result.resid
ar_std = ar_result.conditional_volatility
ar_std_resid = ar_resid /ar_std
plot_acf(ar_std_resid.dropna())

#ljungbox
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(ar_std_resid.dropna() , lags = 12, return_df = True)
print('P-values are: ', lb_test.iloc[0,1])



############################KAMIS
#backtesting untuk train data
me = np.mean(ar_result.resid)             # ME
mae = np.mean(np.abs(ar_result.resid))    # MAE
mse = np.mean((ar_result.resid)**2)       # MSE
rmse = np.mean((ar_result.resid)**2)**.5  # RMSE
print(me,mae,mse,rmse)


#train test
from sklearn.model_selection import train_test_split
from pandas.tseries.offsets import MonthEnd
train,test=train_test_split(results1.resid,test_size=.2,shuffle=False)
len(train)
len(test)


#backtesting untuk test data
#Expanding
dataset=train.copy()
for i in range(14):
    start_loc = 0
    end_loc = len(dataset)
    # Specify expanding rolling window size for model fitting
    ar_gm = arch_model(dataset, p=1, q=0,
                       mean='AR', lags=1, vol='GARCH', dist='skewt')
    gm_result = ar_gm.fit(first_obs = start_loc,
                             last_obs = end_loc)
    # Conduct 1-period mean forecast and save the result
    temp_result = gm_result.forecast(horizon=1).mean
    fcast = temp_result.iloc[-1][0]
    dataset.loc[dataset.index.max() + MonthEnd(1)] = fcast

# Plot the forecast mean
plt.plot(dataset[52:], color='red')
plt.plot(train, color='blue')
plt.plot(test, color='green')
plt.show()

me = np.mean(dataset[52:] - test)             # ME
mae = np.mean(dataset[52:] - test)            # MAE
mse = np.mean((dataset[52:] - test)**2)       # MSE
rmse = np.mean((dataset[52:] - test)**2)**.5  # RMSE
print(me,mae,mse,rmse)

#atau Expanding
ar_gm = arch_model(train, p = 1, q = 0,
                      mean = 'AR',lags=1, vol = 'GARCH', dist = 'skewt')
ar_result=ar_gm.fit()
ar_forecast = ar_result.forecast(horizon = 14).mean

#bukti
plt.plot(dataset[52:], color='red')
plt.plot(dataset[52:].index,ar_forecast.iloc[-1])


#final forecast
sarima_pred1 = results1.get_forecast(steps=10)
sarima_mean1 = sarima_pred1.predicted_mean

ar_forecast = ar_result.forecast(horizon = 10)
ar_mean=ar_forecast.mean.iloc[-1]

plt.plot(s2['NTF'])
plt.plot(sarima_mean1)
plt.plot(sarima_mean1.index,ar_mean)
plt.show()

plt.plot(s2['NTF'])
plt.plot(sarima_mean1.index,sarima_mean1 + ar_mean.values)
plt.show()



#tambahan
#white test
results1.summary()
#hetero white test
results1.test_heteroskedasticity("breakvar")
#tvalue
results1.test_heteroskedasticity("breakvar")[0][0]
#pvalue
results1.test_heteroskedasticity("breakvar")[0][1]

#normality JB
results1.test_normality("jarquebera")
#tvalue
results1.test_normality("jarquebera")[0][0]
#pvalue
results1.test_normality("jarquebera")[0][1]

#corr
results1.test_serial_correlation("ljungbox")
#tvalue
results1.test_serial_correlation("ljungbox")[0][0][0]
#pvalue
results1.test_serial_correlation("ljungbox")[0][1][0]

#aic bic
results1.aic
results1.bic


#forecast 10 kedepan
plt.plot(results1.get_forecast(steps=10).predicted_mean)
#predict mulai dari data ke 10
plt.plot(results1.get_prediction(start=10).predicted_mean)
#predict 10 data terkahir
plt.plot(results1.get_prediction(start=-10).predicted_mean)

#pvalue parameter
results1.pvalues


#ML (data pakai NTF series dan sound wav download di https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds)
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#data
df=pd.read_excel('C:/Users/PREDATOR/OneDrive/Dokumen/ts/NTF series (1).xlsx')
df['Date']= pd.to_datetime(df['Date'])
df2=df.groupby([df['Date'].dt.year, df['Date'].dt.month], as_index=False).last()
df.set_index('Date', inplace=True)
df2.set_index('Date', inplace=True)
data = df.groupby([lambda x: x.year, lambda x: x.month]).sum()
data.set_index(df2['NTF'].asfreq('1M').index,inplace=True)


X=data[["Acct"]]
y=data[["NTF"]]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,shuffle=False)


model = LinearRegression()
model.fit(X_train, y_train)

pred=model.predict((X_test))
print(pred)

plt.plot(y_test.index,pred,color='red')
plt.plot(y_test.index,y_test,color='green')
plt.show()

#heartbeat normal or not normal ?
import librosa as lr
from glob import glob

# List all the wav files in the folder
audio_files = glob('C:/Users/PREDATOR/PycharmProjects/Study/heartbeat/set_a' + '/*.wav')

# Read in the ten audio file, create the time array
audio, sfreq = lr.load(audio_files[10])
print(sfreq)
#sfreq 22050, artinya ada 22050 data recorded per sec

#Make time for index in seconds
time = np.arange(0, len(audio)) / sfreq
#atau
time2 = np.arange(audio.shape[-1]) / sfreq

# Plot audio over time
fig, ax = plt.subplots()
ax.plot(time, audio)
ax.set(xlabel='Time (s)', ylabel='Sound Amplitude')
plt.show()


#Explore audio file
print(audio_files[150])
audio1, sfreq1 = lr.load(audio_files[150])
audio2, sfreq2 = lr.load(audio_files[151])
audio3, sfreq3 = lr.load(audio_files[149])
normal=pd.DataFrame({'1':audio1,'2':audio2[:len(audio1)],'3':audio3[:len(audio1)]},index=np.arange(0,len(audio1))/sfreq1)

print(audio_files[101])
audio4, sfreq4 = lr.load(audio_files[101])
audio5, sfreq5 = lr.load(audio_files[102])
audio6, sfreq6 = lr.load(audio_files[100])
abnormal=pd.DataFrame({'4':audio4[:len(audio1)],'5':audio5[:len(audio1)],'6':audio6[:len(audio1)]},index=np.arange(0,len(audio1))/sfreq1)

#cek Plot audio over time
fig, ax = plt.subplots()
ax.plot(np.arange(0,len(audio1))/sfreq1, audio1)
ax.set(xlabel='Time (s)', ylabel='Sound Amplitude')
plt.show()

# Calculate the time array
time = np.arange(normal.shape[0]) / sfreq1

# Stack the normal/abnormal audio so you can loop and plot
stacked_audio = np.hstack([normal, abnormal]).T

fig, axs = plt.subplots(3, 2, figsize=(15, 7))
# Loop through each audio file / ax object and plot
# .T.ravel() transposes the array, then unravels it into a 1-D vector for looping
for iaudio, ax in zip(stacked_audio, axs.T.ravel()):
    ax.plot(time, iaudio)
    ax.set(xlabel='Time (s)', ylabel='Sound Amplitude')
    plt.suptitle('Normal heartbeat                                       Abnormal heartbeat')


# Average across the audio files of each DataFrame
mean_normal = np.mean(normal, axis=1)
mean_abnormal = np.mean(abnormal, axis=1)

# Plot each average over time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
ax1.plot(time, mean_normal)
ax1.set(title="Normal Data")
ax2.plot(time, mean_abnormal)
ax2.set(title="Abnormal Data")
plt.show()


#Coba train test model predict

normal=pd.DataFrame({'1':lr.load(audio_files[150])[0],
                     '2':lr.load(audio_files[151])[0][:len(audio1)],
                     '3':lr.load(audio_files[152])[0][:len(audio1)],
                     '4': lr.load(audio_files[153])[0][:len(audio1)],
                     '5': lr.load(audio_files[154])[0][:len(audio1)],
                     '6': lr.load(audio_files[155])[0][:len(audio1)],
                     '7': lr.load(audio_files[156])[0][:len(audio1)],
                     '8': lr.load(audio_files[157])[0][:len(audio1)],
                     '9': lr.load(audio_files[158])[0][:len(audio1)],
                     '10': lr.load(audio_files[160])[0][:len(audio1)]
                     }
                     ,index=np.arange(0,len(audio1))/sfreq1)


abnormal=pd.DataFrame({'11':lr.load(audio_files[0])[0][:len(audio1)],
                     '12':lr.load(audio_files[1])[0][:len(audio1)],
                     '13':lr.load(audio_files[2])[0][:len(audio1)],
                     '14': lr.load(audio_files[3])[0][:len(audio1)],
                     '15': lr.load(audio_files[103])[0][:len(audio1)],
                     '16': lr.load(audio_files[104])[0][:len(audio1)],
                     '17': lr.load(audio_files[105])[0][:len(audio1)],
                     '18': lr.load(audio_files[106])[0][:len(audio1)],
                     '19': lr.load(audio_files[140])[0][:len(audio1)],
                     '20': lr.load(audio_files[141])[0][:len(audio1)]
                     }
                     ,index=np.arange(0,len(audio1))/sfreq1)

stacked_audio = np.hstack([normal, abnormal]).T
y=np.array(["Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","ABnormal","ABnormal","ABnormal",
   "ABnormal","ABnormal","ABnormal","ABnormal","ABnormal","ABnormal","ABnormal"])
X_train,X_test,y_train,y_test=train_test_split(stacked_audio,y,test_size=.2,shuffle=True)
X_train.shape
y_train.shape
X_test.shape
y_test.shape

from sklearn.svm import LinearSVC

# Initialize and fit the model
model = LinearSVC()
model.fit(X_train,y_train)

# Generate predictions and score them manually
predictions = model.predict(X_test)
print(sum(predictions == y_test) / len(y_test))


#karena data suara terlalu noisy, bisa kita smoothing pake rolling mean
#untuk itu datanya perlu positif semua, karena kalo ada negatif nanti meannya bisa saling menghabiskan
#rectify (kita bikin positif semua pake abs)
audio, sfreq = lr.load(audio_files[10])
time = np.arange(0, len(audio)) / sfreq
fig, ax = plt.subplots()
ax.plot(np.arange(0,len(audio))/sfreq, audio)
ax.set(xlabel='Time (s)', ylabel='Sound Amplitude')
plt.show()

audio_rectified = pd.DataFrame(audio,index=np.arange(0,len(audio))/sfreq).apply(abs)

# Plot the result
audio_rectified.plot(figsize=(10, 5))
plt.show()

# Smooth by applying a rolling mean
audio_rectified_smooth = audio_rectified.rolling(4500).mean()

# Plot the result
audio_rectified_smooth.plot(figsize=(10, 5))
plt.show()

#tapi keliatan kan kalo pake data asli aja datanya terlalu noisy, agak susah buat klasifikasi
#kita bisa extract feature feature lain dari suara tadi kyk max, mean, min
#nanti feature-feature ini bisa di jadiin variable baru buat dimasukin ke machine learning

#coba bikin data
audio1, sfreq1 = lr.load(audio_files[150])
normal=pd.DataFrame({'1':lr.load(audio_files[150])[0],
                     '2':lr.load(audio_files[151])[0][:len(audio1)],
                     '3':lr.load(audio_files[152])[0][:len(audio1)],
                     '4': lr.load(audio_files[153])[0][:len(audio1)],
                     '5': lr.load(audio_files[154])[0][:len(audio1)],
                     '6': lr.load(audio_files[155])[0][:len(audio1)],
                     '7': lr.load(audio_files[156])[0][:len(audio1)],
                     '8': lr.load(audio_files[157])[0][:len(audio1)],
                     '9': lr.load(audio_files[158])[0][:len(audio1)],
                     '10': lr.load(audio_files[160])[0][:len(audio1)]
                     }
                     ,index=np.arange(0,len(audio1))/sfreq1)


abnormal=pd.DataFrame({'11':lr.load(audio_files[0])[0][:len(audio1)],
                     '12':lr.load(audio_files[1])[0][:len(audio1)],
                     '13':lr.load(audio_files[2])[0][:len(audio1)],
                     '14': lr.load(audio_files[3])[0][:len(audio1)],
                     '15': lr.load(audio_files[103])[0][:len(audio1)],
                     '16': lr.load(audio_files[104])[0][:len(audio1)],
                     '17': lr.load(audio_files[105])[0][:len(audio1)],
                     '18': lr.load(audio_files[106])[0][:len(audio1)],
                     '19': lr.load(audio_files[140])[0][:len(audio1)],
                     '20': lr.load(audio_files[141])[0][:len(audio1)]
                     }
                     ,index=np.arange(0,len(audio1))/sfreq1)

normal=normal.apply(abs)
abnormal=abnormal.apply(abs)
normal=normal.rolling(4500).mean()
abnormal=abnormal.rolling(4500).mean()
stacked_audio = pd.DataFrame(np.hstack([normal, abnormal]))



# Calculate stats
means = np.mean(stacked_audio, axis=0)
stds = np.std(stacked_audio, axis=0)
maxs = np.max(stacked_audio, axis=0)

# Create the X and y arrays
X = np.column_stack([means, stds, maxs])
y=np.array(["Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","ABnormal","ABnormal","ABnormal",
   "ABnormal","ABnormal","ABnormal","ABnormal","ABnormal","ABnormal","ABnormal"])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,shuffle=True)
X_train.shape
y_train.shape
X_test.shape
y_test.shape

# Initialize and fit the model
model = LinearSVC()
model.fit(X_train,y_train)

# Generate predictions and score them manually
predictions = model.predict(X_test)
print(sum(predictions == y_test) / len(y_test))

# Fit the model and score on testing data crossval
from sklearn.model_selection import cross_val_score
percent_score = cross_val_score(LinearSVC(), X, y, cv=5)
print(np.mean(percent_score))

# Calculate the tempo of the
allaudio=pd.DataFrame(stacked_audio.T)
tempos = []
for col, i_audio in allaudio.items():
    tempos.append(lr.beat.tempo(i_audio.values, sr=sfreq, hop_length=2**6, aggregate=None))

# Convert the list to an array so you can manipulate it more easily
tempos = np.array(tempos)

# Calculate statistics of each tempo
tempos_mean = tempos.mean(axis=-1)
tempos_std = tempos.std(axis=-1)
tempos_max = tempos.max(axis=-1)

# Create the X and y arrays
X = np.column_stack([means, stds, maxs, tempos_mean, tempos_std, tempos_max])
y=np.array(["Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","ABnormal","ABnormal","ABnormal",
   "ABnormal","ABnormal","ABnormal","ABnormal","ABnormal","ABnormal","ABnormal"])

# Fit the model and score on testing data
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))


#balik ke NTF, kita coba olah data 2 sekaligus, 1 df (harian), 2 data (agregat bulanan)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
df=pd.read_excel('C:/Users/PREDATOR/OneDrive/Dokumen/ts/NTF series (1).xlsx')
df['Date']= pd.to_datetime(df['Date'])
df2=df.groupby([df['Date'].dt.year, df['Date'].dt.month], as_index=False).last()
df.set_index('Date', inplace=True)
df2.set_index('Date', inplace=True)
data = df.groupby([lambda x: x.year, lambda x: x.month]).sum()
data.set_index(df2['NTF'].asfreq('1M').index,inplace=True)
df=df.reindex(pd.bdate_range('2017-01-03', '2022-06-30'))

#plot
#plot df
fig, ax = plt.subplots()
ax.plot(df.index,df['Acct'])
ax.set(xlabel='Date', ylabel='Acct')
plt.show()
#zoom
plt.plot(df['Acct'][:365])

fig, ax = plt.subplots()
ax.plot(df.index,df['NTF'])
ax.set(xlabel='Date', ylabel='NTF')
plt.show()
#zoom
plt.plot(df['NTF'][:365])


fig, ax = plt.subplots()
ax.plot(df.index,df['Average NTF'])
ax.set(xlabel='Date', ylabel='Average NTF')
plt.show()
#zoom
plt.plot(df['Average NTF'][:365])

#plot data
fig, ax = plt.subplots()
ax.plot(data.index,data['Acct'])
ax.set(xlabel='Date', ylabel='Acct')
plt.show()

fig, ax = plt.subplots()
ax.plot(data.index,data['NTF'])
ax.set(xlabel='Date', ylabel='NTF')
plt.show()

fig, ax = plt.subplots()
ax.plot(data.index,data['Average NTF'])
ax.set(xlabel='Date', ylabel='Avg NTF')
plt.show()

#scatter df
df.plot.scatter("Acct", "NTF")
plt.show()
# Scatterplot with color relating to time
df.plot.scatter('Acct', 'NTF', c=df.index,
                    cmap=plt.cm.viridis, colorbar=False)
plt.show()


#scatter data
data.plot.scatter("Acct", "NTF")
plt.show()
# Scatterplot with color relating to time
data.plot.scatter('Acct', 'NTF', c=data.index,
                    cmap=plt.cm.viridis, colorbar=False)
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
# Use stock symbols to extract training data
X = data[['Acct']]
y = data[['NTF']]

# Fit and score the model with cross-validation
scores = cross_val_score(LinearRegression(), X, y, cv=3)
print(scores)
scores = cross_val_score(Ridge(), X, y, cv=3)
print(scores)

from sklearn.metrics import r2_score

# Split our data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=.8, shuffle=False)

# Fit our model and generate predictions Linear
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)
print(score)

# Fit our model and generate predictions RIDGE
model = Ridge()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)
print(score)

# Visualize our predictions along with the "true" values, and print the score
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(y_test, color='k', lw=3)
ax.plot(y_test.index,predictions, color='r', lw=2)
plt.show()


# Count the missing values of each time series
missing_values = df.isna().sum()
print(missing_values)


# Create a function we'll use to interpolate and plot
def interpolate_and_plot(datas, interpolation):

    # Interpolate the missing values
    datas_interp = datas.interpolate(interpolation)

    # Plot the results, highlighting the interpolated values in black
    fig, ax = plt.subplots(figsize=(10, 5))
    datas_interp.plot(color='r', alpha=.6, ax=ax, legend=False)

    # Now plot the interpolated values on top in red
    datas.plot(ax=ax, color='k', lw=1.5, legend=False)
    plt.show()

# Interpolate using the latest non-missing value
interpolation_type = "zero"
interpolate_and_plot(df['NTF'], interpolation_type)

# Interpolate linearly
interpolation_type = "linear"
interpolate_and_plot(df['NTF'], interpolation_type)

# Interpolate with a quadratic function
interpolation_type = "quadratic"
interpolate_and_plot(df['NTF'], interpolation_type)

df_NTF_interp = df['NTF'].interpolate('linear')

# Your custom function
def percent_change(series):
    # Collect all *but* the last value of this window, then the final value
    previous_values = series[:-1]
    last_value = series[-1]

    # Calculate the % difference between the last value and the mean of earlier values
    percent_change = (last_value - np.mean(previous_values)) / np.mean(previous_values)
    return percent_change

# Apply your custom function and plot
df_perc = df_NTF_interp.rolling(30).apply(percent_change)
df_perc.plot()
plt.show()


def replace_outliers(series):
    # Calculate the absolute difference of each timepoint from the series mean
    absolute_differences_from_mean = np.abs(series - np.mean(series))

    # Calculate a mask for the differences that are > 3 standard deviations from zero
    this_mask = absolute_differences_from_mean > (np.std(series) * 3)

    # Replace these values with the median accross the data
    series[this_mask] = np.nanmedian(series)
    return series


# Apply your preprocessing function to the timeseries and plot the results
prices_perc = replace_outliers(df_perc)
prices_perc.plot()
plt.show()

# Define a rolling window with Pandas, excluding the right-most datapoint of the window
df_perc_rolling = df_perc.rolling(20, min_periods=5, closed='right')

# Define the features you'll calculate for each window
features_to_calculate = [np.min, np.max, np.mean, np.std]

# Calculate these features for your rolling window object
features = df_perc_rolling.aggregate(features_to_calculate)

# Plot the results
ax = features.plot()
df_perc.plot(ax=ax, color='k', alpha=.2, lw=3)
ax.legend(loc=(1.01, .6))
plt.show()

# Import partial from functools
from functools import partial
percentiles = [1, 10, 25, 50, 75, 90, 99]

# Use a list comprehension to create a partial function for each quantile
percentile_functions = [partial(np.percentile, q=percentile) for percentile in percentiles]

# Calculate each of these quantiles on the data using a rolling window
df_perc_rolling = df_perc.rolling(20, min_periods=5, closed='right')
features_percentiles = df_perc_rolling.aggregate(percentile_functions)

# Plot a subset of the result
ax = features_percentiles.plot(cmap=plt.cm.viridis)
ax.legend(percentiles, loc=(1.01, .5))
plt.show()

# Extract date features from the data, add them as columns
df_perc_frame=pd.DataFrame(df_perc)
df_perc_frame['day_of_week'] = df_perc.index.dayofweek
df_perc_frame['week_of_month'] = df_perc.index.weekofyear
df_perc_frame['month_of_year'] = df_perc.index.month

# Print prices_perc
print(df_perc_frame)


# These are the "time lags"
shifts = np.arange(1, 11).astype(int)

# Use a dictionary comprehension to create name: value pairs, one pair per shift
shifted_data = {"lag_{}_day".format(day_shift): df_perc.shift(day_shift) for day_shift in shifts}

# Convert into a DataFrame for subsequent use
df_perc_shifted = pd.DataFrame(shifted_data)

# Plot
ax = df_perc_shifted.plot(cmap=plt.cm.viridis)
df_perc.plot(color='r', lw=2)
ax.legend(loc='best')
plt.show()

# Replace missing values with the median for each column
X = df_perc_shifted.fillna(np.nanmedian(df_perc_shifted))
y = df_perc.fillna(np.nanmedian(df_perc))

# Fit the model
model = Ridge()
model.fit(X, y)



def visualize_coefficients(coefs, names, ax):
    # Make a bar plot for the coefficients, including their names on the x-axis
    ax.bar(names, coefs)
    ax.set(xlabel='Coefficient name', ylabel='Coefficient value')

    # Set formatting so it looks nice
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    return ax

fig, axs = plt.subplots(2, 1, figsize=(10, 5))
y.plot(ax=axs[0])
visualize_coefficients(model.coef_, df_perc_shifted.columns, ax=axs[1])
plt.show()

# Import TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit

# Create time-series cross-validation object
cv = TimeSeriesSplit(n_splits=10)

# Iterate through CV splits
fig, ax = plt.subplots()
for ii, (tr, tt) in enumerate(cv.split(X, y)):
    # Plot the training data on each iteration, to see the behavior of the CV
    ax.plot(tr, ii + y[tr])

ax.set(title='Training data on each CV iteration', ylabel='CV iteration')
plt.show()

# Generate scores for each split to see how the model performs over time
scores = cross_val_score(model, X, y, cv=cv)
scores
