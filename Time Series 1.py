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

# Web scrap the data
html_data = open("web.txt", "r")
soup = BeautifulSoup(html_data, 'lxml')
table = soup.find_all('tbody')[1]
gme_revenue = pd.DataFrame(columns=["Date", "Revenue"])

# Clean the data
for row in table.find_all('tr'):
    col = row.find_all("td")
    date = col[0].text
    revenue = col[1].text
    gme_revenue = gme_revenue.append({"Date": date, "Revenue": revenue}, ignore_index=True)
gme_revenue["Revenue"] = gme_revenue['Revenue'].str.replace(',|\$', "")
gme_revenue.dropna(inplace=True)
gme_revenue = gme_revenue[gme_revenue['Revenue'] != ""]
gme_revenue['Revenue'] = gme_revenue['Revenue'].astype(int)
gme_revenue['Date'] = pd.to_datetime(gme_revenue['Date'])
gme_revenue.set_index('Date', inplace=True)
gme_revenue = gme_revenue.iloc[::-1]
#train,test=train_test_split(gme_revenue['Revenue'],test_size=.2,shuffle=False)

# analyze the data, give rough picture of the data
print(gme_revenue.describe())
gme_revenue.plot()
plt.show()

# Visualize 4 Month Rolling mean adn std
time_series = gme_revenue['Revenue']
print(type(time_series))
time_series.rolling(4).mean().plot(label='4 month rolling mean')
time_series.rolling(4).std().plot(label='4 month rolling std')
time_series.plot()
plt.legend()
plt.show()

# Visualize decompose data
from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(time_series)
decomp.plot()
plt.show()

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
# because the data isn't stationary, we proceed with differencing

# differencing
# first difference
gme_revenue['Revenue Diff 1'] = gme_revenue['Revenue'] - gme_revenue['Revenue'].shift(1)
adf_check(gme_revenue['Revenue Diff 1'].dropna())
gme_revenue['Revenue Diff 1'].plot()
plt.show()
# non-stationary

# second difference
gme_revenue['Revenue Diff 2'] = gme_revenue['Revenue Diff 1'] - gme_revenue['Revenue Diff 1'].shift(1)
adf_check(gme_revenue['Revenue Diff 2'].dropna())
gme_revenue['Revenue Diff 2'].plot()
plt.show()
# stationary

# seasonal difference
gme_revenue['Revenue Diff seasonal'] = gme_revenue['Revenue'] - gme_revenue['Revenue'].shift(4)
adf_check(gme_revenue['Revenue Diff seasonal'].dropna())
gme_revenue['Revenue Diff seasonal'].plot()
plt.show()
# stationary

#heteroscedasticity
import pylab
import scipy.stats as stats
stats.probplot(time_series, dist="norm", plot=pylab)
pylab.show()

#outlier
def outlier_treatment(datacolumn):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn , [25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range,upper_range

lowerbound,upperbound=outlier_treatment(gme_revenue['Revenue'])
print(gme_revenue['Revenue'][(gme_revenue['Revenue'] < lowerbound) | (gme_revenue['Revenue'] > upperbound)])
#gme_revenue.drop(gme_revenue['Revenue'][ (gme_revenue['Revenue']> upperbound) | (gme_revenue['Revenue'] < lowerbound) ].index , inplace=True)

#boxcox
fitteddata,fittedlambda=sc.stats.boxcox(time_series)

fig, ax = plt.subplots(1, 2)

sns.distplot(time_series, hist=False, kde=True,
             label="Non-Normal", ax=ax[0])

sns.distplot(fitteddata, hist=False, kde=True,
             label="Normal",ax=ax[1])
fitteddata=pd.DataFrame(fitteddata)
fitteddata.set_index(time_series.index,inplace=True)
fitteddata.columns=['datatransform']
plt.plot(fitteddata['datatransform'],color='green')
plt.plot(time_series,color='red')

#grangar
from statsmodels.tsa.stattools import grangercausalitytests
gme_revenue['monthtest'] = gme_revenue.index
gme_revenue['monthtest']=gme_revenue['monthtest'].dt.month
grangercausalitytests(gme_revenue[['Revenue', 'monthtest']], maxlag=12)

