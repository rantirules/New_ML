#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas
from pandas import DataFrame 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[20]:


data = pandas.read_csv('cost_revenue_clean.csv')


# In[16]:


data.describe() #tells us the number entries, the mean, standard deviation, quartiles, mean, min, max


# In[17]:


X = DataFrame(data, columns=['production_budget_usd'])
y = DataFrame(data, columns = ['worldwide_gross_usd'])


# In[35]:


plt.figure(figsize=(10,6))
plt.scatter(X, y, alpha = 0.3)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production budget $')
plt.ylabel('Worldwide Gross $')
plt.ylim(0, 3000000000)
plt.xlim(0,450000000)
plt.show()


# In[38]:


regression = LinearRegression()
regression.fit(X,y)


# Slope Coefficient

# In[39]:


regression.coef_ #theta_1


# In[40]:


#intercept
regression.intercept_


# In[ ]:




