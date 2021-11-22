#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


# #### Data Collecting and Processing

# In[2]:


car_dataset = pd.read_csv('car data.csv')
car_dataset.head()


# In[3]:


car_dataset.shape


# In[4]:


car_dataset.info()


# In[5]:


car_dataset.isnull().sum()


# In[6]:


# Checking the distribution of categorical data
print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())


# #### Encoding the Categorical Data

# In[8]:


car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)


# In[9]:


car_dataset.head()


# #### Splitting the data and Target

# In[11]:


X = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
Y = car_dataset['Selling_Price']
print(X)


# In[12]:


print(Y)


# #### Splitting Training and Test Data

# In[13]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state=2)


# #### Model Training

# In[15]:


lin_reg_model=LinearRegression()
lin_reg_model.fit(X_train,Y_train)


# #### Model Evaluation

# In[17]:


training_data_prediction=lin_reg_model.predict(X_train)


# #### R squared Error

# In[18]:


error_score = metrics.r2_score(Y_train,training_data_prediction)
print("R squared Error:",error_score)


# #### Visualize the actual prices and Predicted prices

# In[19]:


plt.scatter(Y_train,training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()


# #### Prediction on Training Data

# In[22]:


test_data_prediction = lin_reg_model.predict(X_test)


# #### R squared error

# In[23]:


error_score = metrics.r2_score(Y_test,test_data_prediction)
print("R squared Error:",error_score)


# In[24]:


plt.scatter(Y_test,test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()


# #### Lasso Regression

# In[25]:


lass_reg_model=Lasso()
lass_reg_model.fit(X_train,Y_train)


# #### Model Evaluation

# In[26]:


training_data_prediction=lass_reg_model.predict(X_train)


# In[28]:


error_score = metrics.r2_score(Y_train,training_data_prediction)
print("R squared Error:",error_score)


# In[29]:


plt.scatter(Y_train,training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()


# In[30]:


test_data_prediction = lass_reg_model.predict(X_test)


# In[31]:


error_score = metrics.r2_score(Y_test,test_data_prediction)
print("R squared Error:",error_score)


# In[32]:


plt.scatter(Y_test,test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()


# In[ ]:




