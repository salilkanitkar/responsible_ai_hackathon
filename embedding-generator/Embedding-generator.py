#!/usr/bin/env python
# coding: utf-8

# # Part 1: Dealing with embeddings

# In[239]:


import datetime
from tensorflow.keras import layers
from tensorflow import feature_column
from sklearn.model_selection import train_test_split
import tensorflow as tf
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


# In[240]:


data_folder = Path("../dataset")
# below paths should be realtive to data_folder
users_file_glob = "AllUsers.csv"
ads_file_glob = "AllAds.csv"
final_dataset = "AllUsers_Ads_Ratings_df.csv"
derived_dataset = "AllUsers_Ads_Ratings_Fav_Unfav_Merged_df.csv"


# In[293]:


df = pd.read_csv(data_folder / f"{final_dataset}")


# In[294]:


# environment settings
# pd.set_option('display.max_column',None)
# pd.set_option('display.max_rows',None)
# pd.set_option('display.max_seq_items',None)
# pd.set_option('display.max_colwidth', 500)
# pd.set_option('expand_frame_repr', True)


# In[295]:


df.head()


# In[296]:


# Merge the cols into one

df["fav"] = df[['fave1', 'fave2', 'fave3', 'fave4', 'fave5']].apply(lambda x: ' '.join(x.map(str)), axis=1)
df["unfav"] = df[['unfave1', 'unfave2', 'unfave3', 'unfave4', 'unfave5']].apply(lambda x: ' '.join(x.map(str)), axis=1)


# In[297]:


# Drop the cols now

favs = ['fave' + str(i) for i in range(1, 11)]
unfavs = ['unfave' + str(i) for i in range(1, 11)]


# In[298]:


for fav in favs:
    df.drop(fav, inplace=True, axis=1)


# In[299]:


for unfav in unfavs:
    df.drop(unfav, inplace=True, axis=1)


# In[300]:


df.sample(10)


# In[301]:


print(df.info())


# In[302]:


# Save it back on disk
df.to_csv(data_folder / f"{derived_dataset}")


# ## Use embeddings

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

# In[303]:


# In[304]:


df = pd.read_csv(data_folder / f"{derived_dataset}")


# Reference -
# https://github.com/balajibalasubramanian/Kaggle-Toxic-Comments-Challenge/blob/master/Keras%20lstm%201%20layer%20%2B%20GloVe%20%2B%20Early%20Stopping%20%2B%20attention%20%2B%20K-fold%20cross-validation.ipynb

# In[305]:


# define text data
df_new = df['fav'].astype(str)


# In[306]:


# initialize the tokenizer
t = Tokenizer()
t.fit_on_texts(df_new)


# In[307]:


vocabulary_list = list(t.word_index .keys())


# In[308]:


vocab_size = len(t.word_index) + 1
vocab_size


# In[309]:


# integer encode the text data
encoded_favs = t.texts_to_sequences(df_new)
encoded_favs


# In[310]:


maxlen = max(len(x) for x in encoded_favs)
maxlen


# In[311]:


# pad the vectors to create uniform length
padded_favs = pad_sequences(encoded_favs, maxlen=maxlen, padding='post')
padded_favs


# In[312]:


chakin.search(lang='English')


# In[313]:


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


# In[314]:


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


# In[315]:


# Test the shape

embedding_matrix.shape
assert vocab_size == embedding_matrix.shape[0], "Vocab size not matching the shape[0] of embeddings"
assert NUMBER_OF_DIMENSIONS == embedding_matrix.shape[1], "Vocab size not matching the shape[1] of embeddings"


# In[316]:


# Check first few items in embedding matrix

count = 0
for word, i in t.word_index.items():
    print(word)
    count = count + 1
    if count == 10:
        break


# # Part 2: Use it in model now in the model

# In[338]:


# In[318]:


# Working on subset of columns(one of each type)


# In[319]:


df.columns


# In[320]:


df_new = df.filter(['Age', 'Gender', 'fav', 'Rating'], axis=1)
df_new.columns


# In[321]:


train, test = train_test_split(df_new, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


# In[322]:


# A utility method to create a tf.data dataset from a Pandas Dataframe

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('Rating')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


# In[323]:


batch_size = 5  # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


# In[ ]:


# ============= START: Try out a demo (ignore this section) =============

# In[324]:


# We will use this batch to demonstrate several types of feature columns
example_batch = next(iter(train_ds))[0]


# In[325]:


example_batch


# In[326]:


# A utility method to create a feature column
# and to transform a batch of data
def demo(feature_column):
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())


# In[327]:


# Numerical column

age = feature_column.numeric_column("Age")
demo(age)


# In[328]:


# Categorical column

gender = feature_column.categorical_column_with_vocabulary_list(
    'Gender', ['M', 'F'])

thal_one_hot = feature_column.indicator_column(gender)
demo(thal_one_hot)


# In[331]:


categorical_voc = tf.feature_column.categorical_column_with_vocabulary_list(key="fav", vocabulary_list=vocabulary_list)

#embedding_initializer = tf.random_uniform_initializer(-1.0, 1.0)
# initializer=embedding_initializer,
# embed_column_dim = math.ceil(len(vocabulary_list) ** 0.25)

embed_column = tf.feature_column.embedding_column(
    categorical_column=categorical_voc,
    dimension=NUMBER_OF_DIMENSIONS,
    trainable=True)

demo(embed_column)


# ============= END : Try out a demo (ignore this section) =============

# In[333]:


feature_columns = []

# numeric cols
for header in ['Age']:
    feature_columns.append(feature_column.numeric_column(header))

# indicator cols
gender = feature_column.categorical_column_with_vocabulary_list(
    'Gender', ['M', 'F'])
gender_one_hot = feature_column.indicator_column(gender)
feature_columns.append(gender_one_hot)

# embedding cols
categorical_voc = tf.feature_column.categorical_column_with_vocabulary_list(key="fav", vocabulary_list=vocabulary_list)

embed_column = tf.feature_column.embedding_column(
    categorical_column=categorical_voc,
    dimension=NUMBER_OF_DIMENSIONS,
    trainable=True)
feature_columns.append(embed_column)


# In[334]:


# Create a feature layer

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


# In[339]:


# Compile and train the model

model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

logdir = Path("logs") / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)
print(f"Logging tensorboard data at {logdir}")

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5,
          callbacks=[tensorboard_callback],
          verbose=2)


# In[348]:


# Load the TensorBoard notebook extension
get_ipython().magic('load_ext tensorboard')
get_ipython().magic('tensorboard --logdir logs/20200426-200936')


# In[340]:


loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)


# In[237]:


df1 = pd.DataFrame([[26, 'M', [5, 22, 5, 22, 57, 82, 43, 13, 146, 147, 82, 43, 13, 57, 82, 43, 13]]])


# In[ ]:


model.predict(df1)


# In[ ]:


# In[ ]:


# In[ ]:


# ### In progress section

# In[177]:


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


# ### References

# In[ ]:


https: // machinelearningmastery.com / use - word - embedding - layers - deep - learning - keras/
https: // machinelearningmastery.com / how - to - prepare - categorical - data - for-deep - learning - in-python/
https: // github.com / guillaume - chevalier / GloVe - as-a - TensorFlow - Embedding - Layer / blob / master / README.md
