from pandas_datareader import data 
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.simplefilter

companies_dict = { #dictionary for extracting company data
    'Amazon':'AMZN',
    'Apple':'AAPL'
}

data_source = 'yahoo'
start_date = '2015-04-25'
end_date = '2020-04-25'
df = data.DataReader(list(companies_dict.values()), data_source, start_date, end_date)

df.head()
df.isna().sum()

stock_open = np.array(df['Open']).T
stock_close = np.array(df['Close']).T
movements = stock_close-stock_open # positive moment implies you can buy the stock
sum_of_movement = np.sum(movements,1)
for i in range(len(companies_dict)):
    print('company:{}, Change:{}'.format(df['High'].columns[i],sum_of_movement))

plt.figure(figsize = (20,10)) 
plt.subplot(1,2,1) 

plt.title('Company:Amazon',fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)
plt.xlabel('Date',fontsize = 15)
plt.ylabel('Opening price',fontsize = 15)
plt.plot(df['Open']['AMZN'])
plt.subplot(1,2,2) 

plt.title('Company:Apple',fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)
plt.xlabel('Date',fontsize = 15)
plt.ylabel('Opening price',fontsize = 15)
plt.plot(df['Open']['AAPL'])

plt.figure(figsize = (20,10)) # Adjusting figure size
plt.title('Company:Amazon',fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)
plt.xlabel('Date',fontsize = 20)
plt.ylabel('Price',fontsize = 20)
plt.plot(df.iloc[0:30]['Open']['AMZN'],label = 'Open') # Opening prices of first 30 days are plotted against date
plt.plot(df.iloc[0:30]['Close']['AMZN'],label = 'Close') # Closing prices of first 30 days are plotted against date
plt.legend(loc='upper left', frameon=False,framealpha=1,prop={'size': 22}) # Properties of legend box