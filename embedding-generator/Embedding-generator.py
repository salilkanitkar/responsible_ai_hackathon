#!/usr/bin/env python
# coding: utf-8

# In[77]:


from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
from collections import defaultdict
import os
import json
import chakin
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pandas as pd
from pathlib import Path


# In[78]:


data_folder = Path("../dataset")
# below paths should be realtive to data_folder
users_file_glob = "AllUsers.csv"
ads_file_glob = "AllAds.csv"
final_dataset = "AllUsers_Ads_Ratings_df.csv"
derived_dataset = "AllUsers_Ads_Ratings_Fav_Unfav_Merged_df.csv"


# In[79]:


df = pd.read_csv(data_folder / f"{users_file_glob}")


# In[108]:


# environment settings
# pd.set_option('display.max_column',None)
# pd.set_option('display.max_rows',None)
# pd.set_option('display.max_seq_items',None)
# pd.set_option('display.max_colwidth', 500)
# pd.set_option('expand_frame_repr', True)


# In[81]:


df.head()


# In[82]:


# Merge the cols into one

df["fav"] = df[['fave1', 'fave2', 'fave3', 'fave4', 'fave5']].apply(lambda x: ' '.join(x.map(str)), axis=1)
df["unfav"] = df[['unfave1', 'unfave2', 'unfave3', 'unfave4', 'unfave5']].apply(lambda x: ' '.join(x.map(str)), axis=1)


# In[83]:


# Drop the cols now

favs = ['fave' + str(i) for i in range(1, 11)]
unfavs = ['unfave' + str(i) for i in range(1, 11)]


# In[84]:


for fav in favs:
    df.drop(fav, inplace=True, axis=1)


# In[ ]:


for unfav in unfavs:
    df.drop(unfav, inplace=True, axis=1)


# In[149]:


df.sample(10)


# In[90]:


# Save it back on disk
df.to_csv(data_folder / f"{derived_dataset}")


# # Use embeddings

# Here we can follow two approaches:<br/>
# -     Use Embedding layer and pass pre trained glove embeddings and fine tune on it (This is what we will try out!)<br/>
# -     Use glove to convert cols to vectors and store it back to disk (Manual way what we did for AI Hackathon)
# -     Encode each column and make it part of the training. Needs separate layer for each cat col in our case fav and unfav
#
#
# Alternatively we can train embeddings on our dataset by using fav and unfav columns and a <br/>
# making a dataset to train a embeddings capturing which words are positive ad which are negative<br/>
# and then use that in our models.
#

# ## Use Embedding layer and pass pre trained glove embeddings

# In[140]:


# In[ ]:


df = pd.read_csv(data_folder / f"{derived_dataset}")


# Reference -
# https://github.com/balajibalasubramanian/Kaggle-Toxic-Comments-Challenge/blob/master/Keras%20lstm%201%20layer%20%2B%20GloVe%20%2B%20Early%20Stopping%20%2B%20attention%20%2B%20K-fold%20cross-validation.ipynb

# In[95]:


# define text data
df_new = df['fav'].astype(str)


# In[97]:


# initialize the tokenizer
t = Tokenizer()
t.fit_on_texts(df_new)
vocab_size = len(t.word_index) + 1


# In[101]:


vocab_size


# In[105]:


# integer encode the text data
encoded_favs = t.texts_to_sequences(df_new)
encoded_favs


# In[119]:


maxlen = max(len(x) for x in encoded_favs)
maxlen


# In[121]:


# pad the vectors to create uniform length
padded_favs = pad_sequences(encoded_favs, maxlen=maxlen, padding='post')
padded_favs


# In[124]:


chakin.search(lang='English')


# In[127]:


# Downloading Twitter.25d embeddings from Stanford:

CHAKIN_INDEX = 17
NUMBER_OF_DIMENSIONS = 25
SUBFOLDER_NAME = "glove.twitter.27B"

DATA_FOLDER = "embeddings"
ZIP_FILE = os.path.join(DATA_FOLDER, "{}.zip".format(SUBFOLDER_NAME))
ZIP_FILE_ALT = "glove" + ZIP_FILE[5:]  # sometimes it's lowercase only...
UNZIP_FOLDER = os.path.join(DATA_FOLDER, SUBFOLDER_NAME)
if SUBFOLDER_NAME[-1] == "d":
    GLOVE_FILENAME = os.path.join(UNZIP_FOLDER, "{}.txt".format(SUBFOLDER_NAME))
else:
    GLOVE_FILENAME = os.path.join(UNZIP_FOLDER, "{}.{}d.txt".format(SUBFOLDER_NAME, NUMBER_OF_DIMENSIONS))


if not os.path.exists(ZIP_FILE) and not os.path.exists(UNZIP_FOLDER):
    # GloVe by Stanford is licensed Apache 2.0:
    #     https://github.com/stanfordnlp/GloVe/blob/master/LICENSE
    #     http://nlp.stanford.edu/data/glove.twitter.27B.zip
    #     Copyright 2014 The Board of Trustees of The Leland Stanford Junior University
    print("Downloading embeddings to '{}'".format(ZIP_FILE))
    chakin.download(number=CHAKIN_INDEX, save_dir='./{}'.format(DATA_FOLDER))
else:
    print("Embeddings already downloaded.")

if not os.path.exists(UNZIP_FOLDER):
    import zipfile
    if not os.path.exists(ZIP_FILE) and os.path.exists(ZIP_FILE_ALT):
        ZIP_FILE = ZIP_FILE_ALT
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        print("Extracting embeddings to '{}'".format(UNZIP_FOLDER))
        zip_ref.extractall(UNZIP_FOLDER)
else:
    print("Embeddings already extracted.")


# In[130]:


# load the glove embedding into memory after downloading and unzippping

embeddings_index = dict()
print("Reading Glove embeddings from ", GLOVE_FILENAME)
f = open(GLOVE_FILENAME, encoding="utf8")

for line in f:
    # Note: use split(' ') instead of split() if you get an error.
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix
embedding_matrix = np.zeros((vocab_size, NUMBER_OF_DIMENSIONS))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[134]:


# Test the shape

embedding_matrix.shape
assert vocab_size == embedding_matrix.shape[0], "Vocab size not matching the shape[0] of embeddings"
assert NUMBER_OF_DIMENSIONS == embedding_matrix.shape[1], "Vocab size not matching the shape[1] of embeddings"


# In[138]:


# Check first few items in embedding matrix

count = 0
for word, i in t.word_index.items():
    print(word)
    count = count + 1
    if count == 10:
        break


# ### Use in the model

# In[142]:


# define model

model = Sequential()
e = Embedding(vocab_size, NUMBER_OF_DIMENSIONS, weights=[embedding_matrix], input_length=maxlen, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))


# In[143]:


# compile the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[144]:


# summarize the model

print(model.summary())


# ### In progress section

# // TODO: Use existing dataset and try a model
#     Include more cols here using TF guide that was shared on slack

# In[ ]:


# fit the model

model.fit(padded_favs, labels, epochs=50, verbose=0)


# In[ ]:


# evaluate the model

loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy * 100))


# In[ ]:


# In[ ]:
