from pandas_datareader import data as pdr
import yfinance as yf 
yf.pdr_override()
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
import plotly.graph_objects as go
import warnings
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
from twilio.rest import Client
from pandas.plotting import table
from IPython.display import display
import dataframe_image as dfi
import streamlit as st
from sklearn.decomposition import PCA


warnings.simplefilter

companies_dict = { #dictionary for extracting company datacd 
    'Amazon':'AMZN',
    'Apple':'AAPL',
    'Walgreen':'WBA',
    'Northrop Grumman':'NOC',
    'Boeing':'BA',
    'Lockheed Martin':'LMT',
    'McDonalds':'MCD',
    'Intel':'INTC',
    'IBM':'IBM',
    'Texas Instruments':'TXN',
    'MasterCard':'MA',
    'Microsoft':'MSFT',
    'General Electrics':'GE',
    'American Express':'AXP',
    'Pepsi':'PEP',
    'Coca Cola':'KO',
    'Johnson & Johnson':'JNJ',
    'Toyota':'TM',
    'Honda':'HMC',
    'Exxon':'XOM',
    'Chevron':'CVX',
    'Valero Energy':'VLO',
    'Ford':'F',
    'Tesla':'TSLA',
    'Bank of America':'BAC'}

start_date = '2017-02-12'
end_date = '2023-02-12'
df = pdr.get_data_yahoo(list(companies_dict.values()), start_date, end_date)

df.head()
df.isna().sum()

stock_open = np.array(df['Open']).T
stock_close = np.array(df['Close']).T
movements = stock_close-stock_open
sum_of_movement = np.sum(movements,1)
for i in range(len(companies_dict)):
    print('company:{}, Change:{}'.format(df['High'].columns[i],sum_of_movement))

# plt.figure(figsize = (20,10)) 
# plt.subplot(1,2,1) 

# plt.title('Company:Amazon',fontsize = 20)
# plt.xticks(fontsize = 10)
# plt.yticks(fontsize = 20)
# plt.xlabel('Date',fontsize = 15)
# plt.ylabel('Opening price',fontsize = 15)
# plt.plot(df['Open']['AMZN'])
# plt.savefig('hack2023/src/my_plot5.png')
# plt.subplot(1,2,2) 

# plt.title('Company:Apple',fontsize = 20)
# plt.xticks(fontsize = 10)
# plt.yticks(fontsize = 20)
# plt.xlabel('Date',fontsize = 15)
# plt.ylabel('Opening price',fontsize = 15)
# plt.plot(df['Open']['AAPL'])
# plt.savefig('hack2023/src/my_plot3.png')


# plt.figure(figsize = (20,10))
# plt.title('Company:Amazon',fontsize = 20)
# plt.xticks(fontsize = 10)
# plt.yticks(fontsize = 20)
# plt.xlabel('Date',fontsize = 20)
# plt.ylabel('Price',fontsize = 20)
# plt.plot(df.iloc[0:30]['Open']['AMZN'],label = 'Open')
# plt.plot(df.iloc[0:30]['Close']['AMZN'],label = 'Close') 
# plt.legend(loc='upper left', frameon=False,framealpha=1,prop={'size': 22}) 

# plt.figure(figsize = (20,10))
# plt.title('Company:Amazon',fontsize = 20)
# plt.xticks(fontsize = 10)
# plt.yticks(fontsize = 20)
# plt.xlabel('Date',fontsize = 20)
# plt.ylabel('Price',fontsize = 20)
# plt.plot(df.iloc[0:30]['Open']['AMZN'],label = 'Open')
# plt.plot(df.iloc[0:30]['Close']['AMZN'],label = 'Close')
# plt.legend(loc='upper left', frameon=False,framealpha=1,prop={'size': 22})

# plt.figure(figsize = (20,8)) 
# plt.title('Company:Amazon',fontsize = 20)
# plt.xticks(fontsize = 18)
# plt.yticks(fontsize = 20)
# plt.xlabel('Date',fontsize = 20)
# plt.ylabel('Movement',fontsize = 20)
# plt.plot(movements[0][0:30])
# plt.savefig('hack2023/src/my_plot6.png')

# plt.figure(figsize = (20,10)) 
# plt.title('Company:Amazon',fontsize = 20)
# plt.xticks(fontsize = 18)
# plt.yticks(fontsize = 20)
# plt.xlabel('Date',fontsize = 20)
# plt.ylabel('Volume',fontsize = 20)
# plt.plot(df['Volume']['AMZN'],label = 'Open')
# plt.savefig('hack2023/src/my_plot4.png')

plt.figure(figsize = (20,8)) 
ax1 = plt.subplot(1,2,1)
plt.title('Company:Tesla',fontsize = 20)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 20)
plt.xlabel('Date',fontsize = 20)
plt.ylabel('Movement',fontsize = 20)
plt.plot(movements[0]) 
plt.subplot(1,2,2,sharey = ax1)
plt.title('Company:Microsoft',fontsize = 20)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 20)
plt.xlabel('Date',fontsize = 20)
plt.ylabel('Movement',fontsize = 20)
plt.plot(movements[1])
plt.savefig('hack2023/src/my_plot3.png')


normalizer = Normalizer()
norm_movements = normalizer.fit_transform(movements)
print(norm_movements.min())
print(norm_movements.max())
print(norm_movements.mean())


normalizer = Normalizer()
kmeans = KMeans(n_clusters = 10,max_iter = 1000)
pipeline = make_pipeline(normalizer,kmeans)
pipeline.fit(movements)
predictions = pipeline.predict(movements)
df1 = pd.DataFrame({'Cluster':predictions,'companies':list(companies_dict)}).sort_values(by=['Cluster'],axis = 0)
#dfi.export(df1, 'hack2023/src/dataframe1.png')


normalizer = Normalizer()
reduced_data = PCA(n_components = 2)
kmeans = KMeans(n_clusters = 10,max_iter = 1000)
pipeline = make_pipeline(normalizer,reduced_data,kmeans)
pipeline.fit(movements)
predictions = pipeline.predict(movements)
df2 = pd.DataFrame({'Cluster':predictions,'companies':list(companies_dict.keys())}).sort_values(by=['Cluster'],axis = 0)
dfi.export(df2, 'hack2023/src/my_plot1.png')

reduced_data = PCA(n_components = 2).fit_transform(norm_movements)
h = 0.01
x_min,x_max = reduced_data[:,0].min()-1, reduced_data[:,0].max() + 1
y_min,y_max = reduced_data[:,1].min()-1, reduced_data[:,1].max() + 1
xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
Z = kmeans.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
cmap = plt.cm.Paired
plt.clf()
plt.figure(figsize=(10,10))
plt.imshow(Z,interpolation = 'nearest',extent=(xx.min(),xx.max(),yy.min(),yy.max()),cmap = cmap,aspect = 'auto',origin = 'lower')
plt.plot(reduced_data[:,0],reduced_data[:,1],'k.',markersize = 5)


centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0],centroids[:,1],marker = 'x',s = 169,linewidths = 3,color = 'w',zorder = 10)
plt.title('K-Means clustering on stock market movements (PCA-Reduced data)')
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.savefig('hack2023/src/my_plot2.png')
