#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pathlib import Path
import re


# In[2]:


data_folder = Path("../dataset")
# below paths should be realtive to data_folder
final_dataset= "users-ads-without-gcp-ratings.csv"
derived_dataset = "users-ads-without-gcp-ratings_OHE.csv"


# In[3]:


df = pd.read_csv(data_folder/f"{final_dataset}")


# In[4]:


# environment settings 
pd.set_option('display.max_column',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_seq_items',None)
pd.set_option('display.max_colwidth', 500)
pd.set_option('expand_frame_repr', True)


# In[5]:


df.head()


# ## OHE Gender

# In[6]:


df.Gender.unique()


# In[7]:


df_Gender = pd.get_dummies(df['Gender'], prefix='Gender')


# In[8]:


df = pd.concat([df, df_Gender], axis=1)


# In[9]:


df.head()


# ## OHE Homecountry

# In[10]:


df.Homecountry.unique()


# In[11]:


df.Homecountry.unique().size


# In[12]:


df_Homecountry = pd.get_dummies(df['Homecountry'], prefix='Homecountry')


# In[13]:


df = pd.concat([df, df_Homecountry], axis=1)


# In[14]:


df = df.rename({
    c: re.sub(r"[^a-zA-Z0-9_]", "", c) for c in df.columns
}, axis=1)


# In[15]:


df.head()


# ## OHE Hometown

# In[16]:


df.Hometown.unique()


# In[17]:


df.Hometown.unique().size


# In[18]:


df_Hometown = pd.get_dummies(df['Hometown'], prefix='Hometown')


# In[19]:


df = pd.concat([df, df_Hometown], axis=1)


# In[20]:


df = df.rename({
    c: re.sub(r"[^a-zA-Z0-9_]", "", c) for c in df.columns
}, axis=1)


# In[21]:


df.head()


# ## OHE Income

# In[22]:


df.Income.unique()


# In[23]:


df.Income.unique().size


# In[24]:


df_Income = pd.get_dummies(df['Income'], prefix='Income')


# In[25]:


df = pd.concat([df, df_Income], axis=1)


# In[26]:


df = df.rename({
    c: re.sub(r"[^a-zA-Z0-9_]", "", c) for c in df.columns
}, axis=1)


# In[27]:


df.head()


# ## Write final CSV

# In[28]:


df.to_csv(data_folder/f"{derived_dataset}", index=False)


# ## Mostlistenedmusics: Embedding + OHE 

# In[61]:


# df.Mostlistenedmusics.unique()


# In[62]:


# ','.join(df.Mostlistenedmusics.unique())


# In[53]:


arr_mostlistenedmusics = ','.join(df.Mostlistenedmusics.unique()).split(",")


# In[54]:


for i, s in enumerate(arr_mostlistenedmusics):
    arr_mostlistenedmusics[i] = s.strip()

arr_mostlistenedmusics


# In[55]:


def get_unique(list1): 
    unique_list = [] 
    for x in list1: 
        if x not in unique_list: 
            unique_list.append(x) 
    return unique_list


# In[57]:


arr_mostlistenedmusics = get_unique(arr_mostlistenedmusics)
arr_mostlistenedmusics


# In[66]:


len(arr_mostlistenedmusics)


# In[ ]:




