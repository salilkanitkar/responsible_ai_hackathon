#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pathlib import Path
import re
from sklearn.preprocessing import MultiLabelBinarizer


# In[2]:


data_folder = Path("../dataset")
# below paths should be realtive to data_folder
final_dataset= "users-ads-without-gcp-ratings_OHE_MLB.csv"
derived_dataset = "users-ads-without-gcp-ratings_OHE_MLB.csv"


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


# ## MLB: Mostlistenedmusics 

# In[7]:


def get_unique(list1): 
    unique_list = [] 
    for x in list1: 
        if x not in unique_list: 
            unique_list.append(x) 
    return unique_list


# In[29]:


arr_mostlistenedmusics = ','.join(df.Mostlistenedmusics.unique()).split(",")

for i, s in enumerate(arr_mostlistenedmusics):
    arr_mostlistenedmusics[i] = s.strip()

arr_mostlistenedmusics = get_unique(arr_mostlistenedmusics)
arr_mostlistenedmusics.sort()
print(len(arr_mostlistenedmusics))


# In[30]:


lst = []
df_Mostlistenedmusics = pd.DataFrame(lst, columns = arr_mostlistenedmusics) 
df_Mostlistenedmusics


# In[31]:


mostlistenedmusics_binarizer = MultiLabelBinarizer()
mostlistenedmusics_binarizer.fit([arr_mostlistenedmusics])


# In[32]:


rowCount = len(df.index)


# In[33]:


for i in range(0, rowCount):
    vals = df.Mostlistenedmusics[i].split(",")
    for index, s in enumerate(vals):
        vals[index] = s.lstrip().rstrip()
    # print(vals)
    rowVals = mostlistenedmusics_binarizer.transform([vals])[0]
    # rowVals
    df_Mostlistenedmusics = df_Mostlistenedmusics.append(pd.Series(rowVals, index=df_Mostlistenedmusics.columns ), ignore_index=True)


# In[34]:


len(df_Mostlistenedmusics.index)


# In[35]:


df_Mostlistenedmusics.head()


# In[36]:


df = pd.concat([df, df_Mostlistenedmusics], axis=1)


# In[37]:


df = df.rename({
    c: re.sub(r"[^a-zA-Z0-9_]", "", c) for c in df.columns
}, axis=1)


# In[38]:


df.head()


# In[ ]:





# ## MLB: Mostreadbooks 

# In[39]:


arr_mostreadbooks = ','.join(df.Mostreadbooks.unique()).split(",")

for i, s in enumerate(arr_mostreadbooks):
    arr_mostreadbooks[i] = s.strip()

arr_mostreadbooks = get_unique(arr_mostreadbooks)
arr_mostreadbooks.sort()
print(len(arr_mostreadbooks))


# In[40]:


lst = []
df_Mostreadbooks = pd.DataFrame(lst, columns = arr_mostreadbooks) 
df_Mostreadbooks


# In[41]:


mostreadbooks_binarizer = MultiLabelBinarizer()
mostreadbooks_binarizer.fit([arr_mostreadbooks])


# In[42]:


rowCount = len(df.index)


# In[43]:


for i in range(0, rowCount):
    vals = df.Mostreadbooks[i].split(",")
    for index, s in enumerate(vals):
        vals[index] = s.lstrip().rstrip()
    # print(vals)
    rowVals = mostreadbooks_binarizer.transform([vals])[0]
    # rowVals
    df_Mostreadbooks = df_Mostreadbooks.append(pd.Series(rowVals, index=df_Mostreadbooks.columns ), ignore_index=True)


# In[44]:


len(df_Mostreadbooks.index)


# In[45]:


df_Mostreadbooks.head()


# In[46]:


df = pd.concat([df, df_Mostreadbooks], axis=1)


# In[47]:


df = df.rename({
    c: re.sub(r"[^a-zA-Z0-9_]", "", c) for c in df.columns
}, axis=1)


# In[48]:


df.head()


# ## MLB: Mostwatchedmovies 

# In[8]:


arr_mostwatchedmovies = ','.join(df.Mostwatchedmovies.unique()).split(",")

for i, s in enumerate(arr_mostwatchedmovies):
    arr_mostwatchedmovies[i] = s.strip()

arr_mostwatchedmovies = get_unique(arr_mostwatchedmovies)
arr_mostwatchedmovies.sort()
print(len(arr_mostwatchedmovies))


# In[9]:


lst = []
df_Mostwatchedmovies = pd.DataFrame(lst, columns = arr_mostwatchedmovies) 
df_Mostwatchedmovies


# In[10]:


mostwatchedmovies_binarizer = MultiLabelBinarizer()
mostwatchedmovies_binarizer.fit([arr_mostwatchedmovies])


# In[11]:


rowCount = len(df.index)


# In[12]:


for i in range(0, rowCount):
    vals = df.Mostwatchedmovies[i].split(",")
    for index, s in enumerate(vals):
        vals[index] = s.lstrip().rstrip()
    # print(vals)
    rowVals = mostwatchedmovies_binarizer.transform([vals])[0]
    # rowVals
    df_Mostwatchedmovies = df_Mostwatchedmovies.append(pd.Series(rowVals, index=df_Mostwatchedmovies.columns ), ignore_index=True)


# In[13]:


len(df_Mostwatchedmovies.index)


# In[14]:


df_Mostwatchedmovies = df_Mostwatchedmovies.add_prefix('Mostwatchedmovies_')
df_Mostwatchedmovies.head()


# In[15]:


df = pd.concat([df, df_Mostwatchedmovies], axis=1)


# In[16]:


df = df.rename({
    c: re.sub(r"[^a-zA-Z0-9_]", "", c) for c in df.columns
}, axis=1)


# In[17]:


df.head()


# ## MLB: Mostwatchedtvprogrammes 

# In[18]:


arr_mostwatchedtvprogrammes = ','.join(df.Mostwatchedtvprogrammes.unique()).split(",")

for i, s in enumerate(arr_mostwatchedtvprogrammes):
    arr_mostwatchedtvprogrammes[i] = s.strip()

arr_mostwatchedtvprogrammes = get_unique(arr_mostwatchedtvprogrammes)
arr_mostwatchedtvprogrammes.sort()
print(len(arr_mostwatchedtvprogrammes))


# In[19]:


lst = []
df_Mostwatchedtvprogrammes = pd.DataFrame(lst, columns = arr_mostwatchedtvprogrammes) 
df_Mostwatchedtvprogrammes


# In[20]:


mostwatchedtvprogrammes_binarizer = MultiLabelBinarizer()
mostwatchedtvprogrammes_binarizer.fit([arr_mostwatchedtvprogrammes])


# In[21]:


rowCount = len(df.index)


# In[22]:


for i in range(0, rowCount):
    vals = df.Mostwatchedtvprogrammes[i].split(",")
    for index, s in enumerate(vals):
        vals[index] = s.lstrip().rstrip()
    # print(vals)
    rowVals = mostwatchedtvprogrammes_binarizer.transform([vals])[0]
    # rowVals
    df_Mostwatchedtvprogrammes = df_Mostwatchedtvprogrammes.append(pd.Series(rowVals, index=df_Mostwatchedtvprogrammes.columns ), ignore_index=True)


# In[23]:


len(df_Mostwatchedtvprogrammes.index)


# In[24]:


df_Mostwatchedtvprogrammes = df_Mostwatchedtvprogrammes.add_prefix('Mostwatchedtvprogrammes_')
df_Mostwatchedtvprogrammes.head()


# In[25]:


df = pd.concat([df, df_Mostwatchedtvprogrammes], axis=1)


# In[26]:


df = df.rename({
    c: re.sub(r"[^a-zA-Z0-9_]", "", c) for c in df.columns
}, axis=1)


# In[27]:


df.head()


# ## MLB: Mostvisitedwebsites 

# In[ ]:


df.Mostvisitedwebsites.unique()


# In[ ]:





# ## Write final CSV

# In[28]:


df.to_csv(data_folder/f"{derived_dataset}", index=False)


# In[ ]:




