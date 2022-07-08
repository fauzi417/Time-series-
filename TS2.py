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
