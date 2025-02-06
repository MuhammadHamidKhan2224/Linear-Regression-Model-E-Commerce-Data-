#!/usr/bin/env python
# coding: utf-8

# # ------------- Linear Regeession Model for the Prediction of 'Yearly Amount Spent' by Customer for Clothing Brank ------------------

# # Importing Necessary Libraries 

# In[46]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv


# #### In the biggening I was facing a issue while importing the dataset because on kaggle it was in txt formate. I almost tried for 30 minutes on this issue. But later got an idea, that similar data must someone has already used and have uploaded project on his/her GitHub. So, finally I downloaded .csv formate from GitHub.

# In[60]:


ecom = pd.read_csv('Ecommerce_Customers.csv')
ecom.head()


# # Number of Rows and Columns in the Dataset

# In[52]:


ecom.shape


# # Checking the Columns Names 

# In[67]:


ecom.columns


# # Checking the Null Values in the Dataset

# In[70]:


ecom.isnull().sum()


# # Some Basic Information About the Dataset

# In[73]:


ecom.info()


# # Some Basic Descriptive Statistics About the Dataset

# ### Numerical Data

# In[95]:


ecom.describe()


# ### Text Data

# In[90]:


ecom.describe(include = 'object').T


# # Exploratory Data Analysis

# ## Our Target Variable is "Yearly Amount Spent"

# ###  Plot 01: J̳o̳i̳n̳t̳p̳l̳o̳t̳ visulize the relationship between two variables along with their individual distribution. It combines scatter plot with histogram. It helps to understand correlation and distribution simultainously. 

# ### Jointplot on Time on Website' vs 'Yearly Amount Spent'

# In[109]:


sns.jointplot(x = 'Time on Website', y = 'Yearly Amount Spent', data=ecom, alpha = 0.5)
plt.show()


# #### Here, in this above plot each point represents a customer, showing their time spent on the website vs. their yearly spending. The scatter appears widely distributed, indicating no clear linear relationship between these variables. It means that people time spent on website not turning to be a costumer. 

# ### Joinplot on 'Time on App' vs 'Yearly Amount Spent'

# In[116]:


sns.jointplot(x = 'Time on App', y = 'Yearly Amount Spent', data=ecom, alpha = 0.5)
plt.show()


# #### Here, in this above plot each point represents a customer, showing their time spent on the app vs. their yearly spending. The scatter plot shows a faint linearity in the relationshio, indicating that their is somehow clear linear relationship between these variables. It means that people time spent on app nearly turning to be a costumer. Again, yet this is not final verdict there is a need to more comprehensive analysis.

# ### Plot 02: Pair Plot
# #### A Pairplot in python is a way to visulize the pairwise relationship between numerical variables in a dataset.
# #### 01: Scatter plot for pairwise relationship
# #### 02: Diagonal of plot show distribution usually contians histograms
# #### 03: Usefull for correlation and trends
# 

# In[135]:


sns.pairplot(ecom, kind = 'scatter', plot_kws = {'alpha': 0.6})
plt.show()


# #### So, this is bit confusing. But we are only interested in last row and last column among all these multiple plots at the time. But this is giving some general idea of the datapoint distribution and to understand their relationship. The glaring example of a clear Linear Relationship of 'Length of Membership' vs 'Year Amount Spent' 

# ### Plot 03: Implot 
# #### It is used to plot the Linear Regression Model for visulizaing relationship between the two variables.

# In[146]:


sns.lmplot( x= 'Length of Membership', y = 'Yearly Amount Spent', data = ecom ,scatter_kws={'alpha' : 0.4})
plt.show()


# #### Conceptually, Simple Linear Regression Model is basically of two variable. One variable is Independent and another is Dependent. Here, above plot is show the nearly strong relationship between 'Length of Membership' and 'Yearly Amount Spent'. It basically drawn this bet fit line using Gradient Descent Algorithum. This algorithum reiteratively improves the model performance untill it achives the MSE minimum possible. In simple words, there is a minimal difference between actual value and predicted value.

# # Importing sklearn Library 

# #### We will divide our imported data into training set and test set. It is general principle in machine learning. Common practice is 80% for training and 20% for test. 

# In[153]:


from sklearn.model_selection import train_test_split


# In[169]:


x = ecom[['Time on App','Time on Website','Length of Membership','Avg. Session Length']]
y = ecom['Yearly Amount Spent']


# In[171]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)


# In[173]:


x_train


# In[175]:


y_train


# In[177]:


x_test


# In[179]:


y_test


# #### Yes you have correctly noticed that the random index numbers in all four. It the buity of the Sklearn library that it randomly select dataset for train and test spliting. 

# # Traning the Model

# In[184]:


from sklearn.linear_model import LinearRegression


# In[186]:


lm = LinearRegression()


# # Fiting the Model

# In[190]:


lm.fit(x_train, y_train)


# # Coefficient of the Model

# #### Coefficient of the model is basically a value of y (DV) when x (IV) is equal to zero. It guides about the importance of the IV, it has direct association like high coef mean highly important.  It the additive factor in the rate of change (slop m) of the linear regression model straight line.
# #### The more variables we have the more coefficients we have.

# In[194]:


lm.coef_


# In[202]:


cdf = pd.DataFrame(lm.coef_,x.columns, columns=['coef'])
print(cdf)


# #### Here, see Length of Membership  61.896829 it has comparatively high coefficient, so it has high correlation, and accordingly other variables will be treated as follows. 

# # Prediction 

# In[207]:


prd = lm.predict(x_test)


# In[209]:


prd


# # Scatter Plot 
# ## This scatter plot is comparing the data points of predicted values and actual values. 

# In[220]:


sns.scatterplot(x=prd, y=y_test)  
plt.xlabel('Predictions')
plt.title('Evaluation of Linear Regression Model')
plt.show()


# ##### As the scatter plot shows, this model behaving seems good, and predicted the values accuratly. But for clarity lets make other comprehensive calculations.

# # Drawing an Evaluation Matrix

# In[225]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


# ### Mean Squared Error

# In[231]:


print('Mean Squared Error: ' , mean_squared_error(y_test, prd))
print('Mean Absolute Error: ' , mean_absolute_error(y_test, prd))
print('RMSE: ' ,math.sqrt(mean_squared_error(y_test, prd))) 


# ### This model seems to be prity good and predictions.

# # Residuals
# ### Residuals are basically the error term ( Difference between predicted value and actual value). In Linear Regression it is primarly assumed that Residuals are normally distributed , mean it has bell-shaped curve in the density plot. In linear regression these residuls are supposed to be random. If these are not random, so its means their is something scarry about the model, and it indicates some baisness in the model.

# In[237]:


residuals = y_test - prd


# In[239]:


print(residuals)


# In[254]:


sns.displot(residuals, bins = 20, kde = True)
plt.show()


# ## This show  almost a normal distribution. 

# # QQ Plot
# ### QQ plot basically plot the normality on one side and the residuls on the other side. It it creates straight line it means residulas are normally distributed.

# In[264]:


import pylab
import scipy.stats as stats
stats.probplot(residuals, dist = 'norm', plot = pylab)
pylab.show()


# ## There it go, here it shows the almost straight line. There is very much minimal difference in actual value and predicted value. 

# # In a nutshell, model in quite confident and well trained to have a good prediction. Let's apply this to get benefit out of it. This model in available on very economic charges. Please feel free to reach out to me anytime.

# In[ ]:




