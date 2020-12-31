#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import os
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import mean_squared_error
from math import sqrt
from statistics import mean 

import warnings
warnings.filterwarnings("ignore")


# In[2]:


os.chdir('./data')


# In[3]:


os.getcwd()


# In[4]:


df = pd.read_csv("energy_consumption.csv")


# In[5]:


df


# In[6]:


df.dtypes


# In[7]:


df.columns


# In[8]:


df.DATE


# In[9]:


df.shape


# In[10]:


df.head(25)


# In[11]:


str('01/') + str(df.DATE[0])


# In[12]:


df['actual_date'] = df['DATE'].apply(lambda x: str('01/') + str(x))


# In[13]:


df.head(25)


# In[14]:


df.dtypes


# In[15]:


df['actual_date'] = pd.to_datetime(df['actual_date'],format='%d/%m/%Y')


# In[16]:


df.head()


# In[17]:


df.index = df.actual_date


# In[18]:


df


# In[19]:


df.drop({'DATE','actual_date'},axis=1, inplace=True)


# In[20]:


df.isnull().sum()


# In[21]:


df


# In[22]:


plt.figure(figsize=(12,8))
df['ENERGY_INDEX'].plot()


# In[23]:


from statsmodels.tsa.seasonal import seasonal_decompose

decomposed_series = seasonal_decompose(df['ENERGY_INDEX'],model='multiplicative')


# In[24]:


# print(decomposed_series.trend)
print(decomposed_series.seasonal)
# print(decomposed_series.resid)
# print(decomposed_series.observed)


# In[25]:


decomposed_series.plot()
plt.show()


# In[59]:


(decomposed_series.seasonal[0:12]).plot(figsize=(20,10), linewidth=5, fontsize=20)


# In[27]:


df.index.min(), df.index.max()


# In[28]:


#Diff
df.index.max() - df.index.min()


# In[29]:


#Holt Winters (Triple exponential smoothening)
from statsmodels.tsa.api import ExponentialSmoothing


# In[30]:


df


# In[31]:


df.shape


# In[32]:


train_data = df[:726]
valid_data = df[726:]


# In[33]:


train_data.shape


# In[34]:


train_data.head()


# In[35]:


train_data.tail()


# In[36]:


valid_data.head()


# In[37]:


valid_data.tail()


# In[38]:


valid_data.shape


# In[39]:


#Exponential smoothening
from statsmodels.tsa.api import SimpleExpSmoothing


# In[40]:


model = SimpleExpSmoothing(np.asarray(train_data['ENERGY_INDEX']))
model = model.fit(smoothing_level=0.7,initial_level=3,optimized=False) 

valid_data['SES'] = model.forecast(len(valid_data)) 


# In[41]:


model.params


# In[42]:


plt.figure(figsize=(12,8))

plt.plot(train_data.index, train_data['ENERGY_INDEX'], label='train_data')
plt.plot(valid_data.index,valid_data['ENERGY_INDEX'], label='valid')
plt.plot(valid_data.index,valid_data['SES'], label='SES Forecast')
plt.legend(loc='best')
plt.title("Simple Exponential Smoothing Method")
plt.show()


# In[43]:


# calculating RMSE 
rmse = sqrt(mean_squared_error(valid_data['ENERGY_INDEX'], valid_data['SES']))
print('The RMSE value for Simple Exponential Smoothing Method is', rmse)


# In[44]:


model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=2, damped=True)
hw_model = model.fit(optimized=True, use_boxcox=False, remove_bias=False)
pred = hw_model.predict(start='2019-09-01', end='2022-09-01')

y_forecast = hw_model.forecast(12)
# rmse = np.sqrt(mean_squared_error(valid_data,y_forecast))
# print(rmse)


# In[45]:


plt.plot(train_data.index, train_data, label='Train data set')
plt.plot(valid_data.index, valid_data, label='Validation set')
plt.plot(pred.index, pred, label='Holt-Winters')
plt.legend(loc='best');


# In[46]:


train_data.index.max(), train_data.index.min()


# In[47]:


valid_data.index.min(), valid_data.index.max()


# In[48]:


smoothing_level=0.4
fit1 = SimpleExpSmoothing(train_data['ENERGY_INDEX']).fit(smoothing_level=smoothing_level,optimized=False)
fcast1 = fit1.forecast(36).rename(r'$\alpha={}$'.format(smoothing_level))
# specific smoothing level
fcast1.plot(marker='o', color='blue', legend=True)
fit1.fittedvalues.plot(marker='o',  color='blue')
mse1 = ((fcast1 - valid_data) ** 2).mean()
print('The Root Mean Squared Error of our forecasts with smoothing level of {} is {}'.format(smoothing_level,round(np.sqrt(mse1), 2)))

## auto optimization
fit2 = SimpleExpSmoothing(train_data['ENERGY_INDEX']).fit()
fcast2 = fit2.forecast(36).rename(r'$\alpha=%s$'%fit2.model.params['smoothing_level'])
# plot
fcast2.plot(marker='o', color='green', legend=True)
fit2.fittedvalues.plot(marker='o', color='green')

mse2 = ((fcast2 - valid_data['ENERGY_INDEX']) ** 2).mean()
print('The Root Mean Squared Error of our forecasts with auto optimization is {}'.format(round(np.sqrt(mse2), 2)))

plt.show()


# In[49]:


# Double Exponential Smoothing


# In[50]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[51]:


if 'Holt_Winter' in valid_data.columns:
    valid_data.drop({'Holt_Winter'}, axis=1, inplace=True)
model = ExponentialSmoothing(np.asarray(train_data['ENERGY_INDEX']) ,seasonal_periods=7 ,trend='add', seasonal='mul',)
fit1 = model.fit(smoothing_level=0.4, smoothing_slope=0.1, smoothing_seasonal=0.8) #

fcast = fit1.forecast(len(valid_data) + 36) 
fcast
valid_data['Holt_Winter'] = fit1.forecast(len(valid_data)) 


# In[52]:


model.params


# In[53]:


plt.figure(figsize=(12,8))

plt.plot(train_data.index, train_data['ENERGY_INDEX'], label='train_data')
plt.plot(valid_data.index,valid_data['ENERGY_INDEX'], label='valid')
plt.plot(valid_data.index,valid_data['Holt_Winter'], label='holt winter Forecast')
plt.legend(loc='best')
plt.title("Triple Exponential Smoothing (holt winter) Method")
plt.show()


# In[54]:


valid_data


# In[55]:


valid_data.columns


# In[56]:


def holt_win_sea(y,y_to_train,y_to_test,seasonal_type,seasonal_period,predict_date):
    
    y.plot(marker='o', color='black', legend=True, figsize=(14, 7))
    
    if seasonal_type == 'additive':
        fit1 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='add').fit(use_boxcox=True)
        fcast1 = fit1.forecast(predict_date).rename('Additive')
        mse1 = ((fcast1 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of additive trend, additive seasonal of '+ 
              'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse1), 2)))
        
        fit2 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='add', damped=True).fit(use_boxcox=True)
        fcast2 = fit2.forecast(predict_date).rename('Additive+damped')
        mse2 = ((fcast2 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of additive damped trend, additive seasonal of '+ 
              'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse2), 2)))
        
        fit1.fittedvalues.plot(style='--', color='red')
        fcast1.plot(style='--', marker='o', color='red', legend=True)
        fit2.fittedvalues.plot(style='--', color='green')
        fcast2.plot(style='--', marker='o', color='green', legend=True)
    
    elif seasonal_type == 'multiplicative':  
        fit3 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='mul').fit(use_boxcox=True)
        fcast3 = fit3.forecast(predict_date).rename('Multiplicative')
        mse3 = ((fcast3 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of additive trend, multiplicative seasonal of '+ 
              'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse3), 2)))
        
        fit4 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='mul', damped=True).fit(use_boxcox=True)
        fcast4 = fit4.forecast(predict_date).rename('Multiplicative+damped')
        mse4 = ((fcast3 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of additive damped trend, multiplicative seasonal of '+ 
              'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse4), 2)))
        
        fit3.fittedvalues.plot(style='--', color='red')
        fcast3.plot(style='--', marker='o', color='red', legend=True)
        fit4.fittedvalues.plot(style='--', color='green')
        fcast4.plot(style='--', marker='o', color='green', legend=True)
        
    else:
        print('Wrong Seasonal Type. Please choose between additive and multiplicative')

    plt.show()


# In[61]:


holt_win_sea(df, train_data, valid_data,'multiplicative',3, len(valid_data)+36)


# In[ ]:




