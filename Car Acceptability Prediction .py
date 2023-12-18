#!/usr/bin/env python
# coding: utf-8

# ## 1. Imports

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 2. Loading the data 

# In[2]:


from sklearn.datasets import fetch_openml

car_data = fetch_openml(name = 'car' , version = 2)


# In[5]:


type(car_data)


# In[6]:


car_data.details


# In[7]:


car_data.details['version']


# In[8]:


# Data Description

print(car_data.DESCR)


# In[10]:


# Displaying feature names

car_data.feature_names


# In[11]:


# Getting the whole dataframe

car_data = car_data.frame
car_data.head()


# In[12]:


type(car_data)


# ### 3. Exploratory Data Analysis

# ##### Splitting Data into Training and Test sets

# In[13]:


from sklearn.model_selection import train_test_split

train_data , test_data = train_test_split(car_data , test_size = 0.1 , random_state = 20 )

print('the size of training data is : {} \n The size of testing data is : {}' .format(len(train_data) ,len(test_data)))


# #### Checking Summary Statistics

# In[14]:


train_data.describe()


# #### Checking Missing Values

# In[15]:


# Checking missing values
train_data.isnull().sum()


# ##### Checking Categorical features

# In[16]:


train_data['buying'].value_counts()


# In[17]:


plt.figure(figsize = (15,10))
sns.countplot(data = train_data , x = 'buying')


# In[18]:


plt.figure(figsize = (15,10))
sns.countplot(data = train_data , x = 'buying' , hue = 'binaryClass') 


# In[19]:


train_data['maint'].value_counts()


# In[21]:


plt.figure(figsize = (15,10))
sns.countplot(data = train_data , x = 'maint')


# In[24]:


plt.figure(figsize = (15,10))
sns.countplot(data = train_data , x = 'maint' , hue = 'binaryClass')


# In[26]:


train_data['doors'].value_counts()


# In[27]:


plt.figure(figsize = (15,10))
sns.countplot(data = train_data , x = 'doors')


# In[28]:


plt.figure(figsize = (15,10))
sns.countplot(data = train_data , x = 'doors' , hue = 'binaryClass')


# In[29]:


train_data['persons'].value_counts()


# In[30]:


plt.figure(figsize = (15,10))
sns.countplot(data = train_data , x = 'persons' , hue = 'binaryClass')


# In[31]:


train_data['lug_boot'].value_counts()


# In[32]:


plt.figure(figsize = (15,10))
sns.countplot(data = train_data , x = 'lug_boot' , hue = 'binaryClass') 


# In[33]:


train_data['safety'].value_counts()


# In[34]:


plt.figure(figsize = (15,10))
sns.countplot( data = train_data , x = 'safety' , hue = 'binaryClass')


# In[36]:


train_data['binaryClass'].value_counts()


# In[37]:


plt.figure(figsize = (15,10))
sns.countplot( data = train_data , x = 'binaryClass')


# ###### As you can see that our data is completely skewed / imbalanced . The positive example are 2x more than a negative examples.
# ######  Accuracy is not right metric in this case . Real world datasets comes with their unique blends , dataset can be imbalanced . Missing values can be present . We just have to find effective way to deal with those issues. WE WILL NOT RELY ON ACCURACY.

# ### 4. Data Preprocessing

# ##### Handling Categorical Features

# In[38]:


###### Decision trees don't care if the feature are scaled pr not , and they can handle the categorical features. 


# In[39]:


car_train = train_data.drop('binaryClass' , axis = 1)
car_labels = train_data[['binaryClass']]


# In[40]:


from sklearn.preprocessing import OrdinalEncoder

from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('ord_enc' , OrdinalEncoder())
])

car_train_prepared = pipe.fit_transform(car_train)


# ###### labels contain P and N so we can convert those numbers . Here instead of using Ordinary Encoder , we will use Label Encoder .  sklearn is explicitly that it is used to encode target features

# In[41]:


from sklearn.preprocessing import LabelEncoder

label_enc = LabelEncoder()

car_labels_prepared = label_enc.fit_transform(car_labels)


# # 5. Training Decision Tree Classifier 

# In[42]:


from sklearn.tree import DecisionTreeClassifier


tree_clg = DecisionTreeClassifier()

tree_clg.fit(car_train_prepared , car_labels_prepared)


# # 6. Evaluating Decision Trees

# In[43]:


from sklearn.metrics import accuracy_score

def accuracy(input_data , models , labels) :
    """
    Take the input data , model and labels and return accuracy
    
    """
    
    preds = model.predict(input_data)
    acc = accuracy_score(labels,preds)
    
    
    return acc


# In[45]:


from sklearn.metrics import confusion_matrix


def conf_matrix (input_data , model, labels) :
    """
    Take the input data , model and labels and return confusion matrix
    
    """
    
    preds = sklearn.predict(input_data)
    cm = confusion_matrix(labels,preds)
    
    return cm


# In[47]:


from sklearn.metrics import classification_report

def class_report(input_data , model , labels):
    """
    take the input data, model and labels and return confusion matrix
    
    """
    
    preds = model.predict(input_data)
    report = classification_report(labels,preds)
    report = print(report)
    
    return report


# In[ ]:


accuracy

