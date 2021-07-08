#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from statsmodels.tsa.holtwinters import ExponentialSmoothing,SimpleExpSmoothing, Holt
import statsmodels.api as sm

data = pd.read_csv('PS_Gi_Throughput.csv')
data = data[data['ne_id']=="GGCBT18"]

data['value'] = pd.to_numeric(data['value'])

model_wma = ExponentialSmoothing(data['value'],seasonal='additive',seasonal_periods=48).fit()

pickle.dump(model_wma, open('model.pkl','wb'))


# In[ ]:




