#!/usr/bin/env python
# coding: utf-8

# # Neural Network Model
# 
# The aim of the notebook is demo end to end pipeline for Ads prediction in Tensorflow

# In[1]:


get_ipython().system(' ./setup.sh')


# In[36]:


get_ipython().system('pip install git+https://github.com/tensorflow/docs')


# In[39]:


get_ipython().system('pip3 install -q git+https://github.com/tensorflow/docs --user')


# In[30]:


import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import sys
from pathlib import Path
import datetime
import time
import numpy as np
import pandas as pd
from pprint import pprint
from typing import Dict, Any, Union, List
from functools import partial
import re
import string
from sklearn.preprocessing import MultiLabelBinarizer
from math import ceil
from collections import namedtuple

print(f"Using Tensorflow, {tf.__version__} on Python interpreter, {sys.version_info}")


# In[31]:


RANDOM_SEED = int(time.time())

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"Using random seed, {RANDOM_SEED}")


# ## Load Data
# 
# Dataset credits:
# ```
# @inproceedings{roffo2016personality,
#   title={Personality in computational advertising: A benchmark},
#   author={Roffo, Giorgio and Vinciarelli, Alessandro},
#   booktitle={4 th Workshop on Emotions and Personality in Personalized Systems (EMPIRE) 2016},
#   pages={18},
#   year={2016}
# }
# ```

# In[32]:


DATA_FOLDER = Path("../../dataset/")
BATCH_SIZE = 4096 # bigger the batch, faster the training but bigger the RAM needed
TARGET_COL = "Rating"

# data files path are relative DATA_FOLDER
users_ads_rating_csv = DATA_FOLDER/"users-ads-without-gcp-ratings_OHE_MLB.csv"


# In[33]:


USER_ID = "UserId"
AD_ID = "AdId"
AGE = "Age"
ZIP_CODE = "CapZipCode"
COUNTRIES_VISITED = "Countriesvisited"
FAVE_SPORTS = "FaveSports"
GENDER = "Gender"
HOME_COUNTRY = "Homecountry"
HOME_TOWN = "Hometown"
INCOME = "Income"
LAST_NAME = "LastName"
MOST_LISTENED_MUSICS = "Mostlistenedmusics"
MOST_READ_BOOKS = "Mostreadbooks"
MOST_VISITED_WEBSITES = "Mostvisitedwebsites"
MOST_WATCHED_MOVIES = "Mostwatchedmovies"
MOST_WATCHED_TV_PROGRAMMES = "Mostwatchedtvprogrammes"
NAME = "Name"
PAYPAL = "Paypal"
TIMEPASS = "Timepass"
TYPE_OF_JOB = "TypeofJob"
WEEKLY_WORKING_HOURS = "Weeklyworkinghours"
FAVE1 = "fave1"
FAVE10 = "fave10"
FAVE2 = "fave2"
FAVE3 = "fave3"
FAVE4 = "fave4"
FAVE5 = "fave5"
FAVE6 = "fave6"
FAVE7 = "fave7"
FAVE8 = "fave8"
FAVE9 = "fave9"
UNFAVE1 = "unfave1"
UNFAVE2 = "unfave2"
UNFAVE3 = "unfave3"
UNFAVE4 = "unfave4"
UNFAVE5 = "unfave5"
UNFAVE6 = "unfave6"
ADFILEPATH = "AdFilePath"
GENDER_F = "Gender_F"
GENDER_M = "Gender_M"
# HomeCountry = 12 Columns
HOMECOUNTRY_CANADA = "Homecountry_Canada"
HOMECOUNTRY_CZECHREPUBLIC = "Homecountry_CzechRepublic"
HOMECOUNTRY_GREATBRITAIN = "Homecountry_GreatBritain"
HOMECOUNTRY_INDIA = "Homecountry_India"
HOMECOUNTRY_ITALY = "Homecountry_Italy"
HOMECOUNTRY_PHILLIPINES = "Homecountry_Phillipines"
HOMECOUNTRY_ROMANIA = "Homecountry_Romania"
HOMECOUNTRY_SAUDIARABIA = "Homecountry_SaudiArabia"
HOMECOUNTRY_SINGAPORE = "Homecountry_Singapore"
HOMECOUNTRY_SLOVENIA = "Homecountry_Slovenia"
HOMECOUNTRY_UNITEDKINGDOM = "Homecountry_UnitedKingdom"
HOMECOUNTRY_UNITEDSTATESOFAMERICA = "Homecountry_UnitedStatesofAmerica"
# Income = 4 Columns
INCOME_0 = "Income_0"
INCOME_1 = "Income_1"
INCOME_2 = "Income_2"
INCOME_3 = "Income_3"
RATING = "Rating"
AD_NUM_FACES = "ad_num_faces"
AD_LABEL_FEATURE_1 = 'ad_isAdvertising'
AD_LABEL_FEATURE_2 = 'ad_isBrand'
AD_LABEL_FEATURE_3 = 'ad_isElectronic device'
AD_LABEL_FEATURE_4 = 'ad_isElectronics'
AD_LABEL_FEATURE_5 = 'ad_isFashion accessory'
AD_LABEL_FEATURE_6 = 'ad_isFictional character'
AD_LABEL_FEATURE_7 = 'ad_isFont'
AD_LABEL_FEATURE_8 = 'ad_isFurniture'
AD_LABEL_FEATURE_9 = 'ad_isGadget'
AD_LABEL_FEATURE_10 = 'ad_isGames'
AD_LABEL_FEATURE_11 = 'ad_isGraphic design'
AD_LABEL_FEATURE_12 = 'ad_isGraphics'
AD_LABEL_FEATURE_13 = 'ad_isJewellery'
AD_LABEL_FEATURE_14 = 'ad_isLine'
AD_LABEL_FEATURE_15 = 'ad_isLogo'
AD_LABEL_FEATURE_16 = 'ad_isMagenta'
AD_LABEL_FEATURE_17 = 'ad_isMaterial property'
AD_LABEL_FEATURE_18 = 'ad_isMultimedia'
AD_LABEL_FEATURE_19 = 'ad_isProduct'
AD_LABEL_FEATURE_20 = 'ad_isRectangle'
AD_LABEL_FEATURE_21 = 'ad_isSkin'
AD_LABEL_FEATURE_22 = 'ad_isTechnology'
AD_LABEL_FEATURE_23 = 'ad_isText'
AD_LABEL_FEATURE_24 = 'ad_isVehicle'
AD_LABEL_FEATURE_25 = 'ad_isYellow'
AD_SAFESEARCH_FEATURE_1 = 'ad_isAdult_UNLIKELY'
AD_SAFESEARCH_FEATURE_2 ='ad_isAdult_VERY_UNLIKELY'
AD_SAFESEARCH_FEATURE_3 ='ad_isSpoof_POSSIBLE'
AD_SAFESEARCH_FEATURE_4 ='ad_isSpoof_UNLIKELY'
AD_SAFESEARCH_FEATURE_5 ='ad_isSpoof_VERY_UNLIKELY'
AD_SAFESEARCH_FEATURE_6 ='ad_isMedical_POSSIBLE'
AD_SAFESEARCH_FEATURE_7 ='ad_isMedical_UNLIKELY'
AD_SAFESEARCH_FEATURE_8 ='ad_isMedical_VERY_UNLIKELY'
AD_SAFESEARCH_FEATURE_9 ='ad_isViolence_VERY_UNLIKELY'
AD_SAFESEARCH_FEATURE_10 ='ad_isRacy_POSSIBLE'
AD_SAFESEARCH_FEATURE_11 ='ad_isRacy_UNLIKELY'
AD_SAFESEARCH_FEATURE_12 ='ad_isRacy_VERY_LIKELY'
AD_SAFESEARCH_FEATURE_13 ='ad_isRacy_VERY_UNLIKELY'

# Read all columns as strings to avoid any errors
COL_DEFAULTS = {
    USER_ID: "**",
    AD_ID: "**",
    AGE: "**",
    ZIP_CODE: "**",
    COUNTRIES_VISITED: "**",
    FAVE_SPORTS: "**",
    GENDER: "**",
    HOME_COUNTRY: "**",
    HOME_TOWN: "**",
    INCOME: "**",
    LAST_NAME: "**",
    MOST_LISTENED_MUSICS: "**",
    MOST_READ_BOOKS: "**",
    MOST_VISITED_WEBSITES: "**",
    MOST_WATCHED_MOVIES: "**",
    MOST_WATCHED_TV_PROGRAMMES: "**",
    NAME: "**",
    PAYPAL: "**",
    TIMEPASS: "**",
    TYPE_OF_JOB: "**",
    WEEKLY_WORKING_HOURS: "**",
    FAVE1: "**",
    FAVE10: "**",
    FAVE2: "**",
    FAVE3: "**",
    FAVE4: "**",
    FAVE5: "**",
    FAVE6: "**",
    FAVE7: "**",
    FAVE8: "**",
    FAVE9: "**",
    UNFAVE1: "**",
    UNFAVE2: "**",
    UNFAVE3: "**",
    UNFAVE4: "**",
    UNFAVE5: "**",
    UNFAVE6: "**",
    ADFILEPATH: "**",
    GENDER_F: "**",
    GENDER_M: "**",
    HOMECOUNTRY_CANADA: "**",
    HOMECOUNTRY_CZECHREPUBLIC: "**",
    HOMECOUNTRY_GREATBRITAIN: "**",
    HOMECOUNTRY_INDIA: "**",
    HOMECOUNTRY_ITALY: "**",
    HOMECOUNTRY_PHILLIPINES: "**",
    HOMECOUNTRY_ROMANIA: "**",
    HOMECOUNTRY_SAUDIARABIA: "**",
    HOMECOUNTRY_SINGAPORE: "**",
    HOMECOUNTRY_SLOVENIA: "**",
    HOMECOUNTRY_UNITEDKINGDOM: "**",
    HOMECOUNTRY_UNITEDSTATESOFAMERICA: "**",
    INCOME_0: "**",
    INCOME_1: "**",
    INCOME_2: "**",
    INCOME_3: "**",
    RATING: "**",
    AD_NUM_FACES: "**"
}

# SELECTED_COLS = [AGE, ZIP_CODE, FAVE_SPORTS, GENDER, HOME_COUNTRY, HOME_TOWN, INCOME, MOST_LISTENED_MUSICS, MOST_READ_BOOKS, 
#                  MOST_VISITED_WEBSITES, MOST_WATCHED_MOVIES, MOST_WATCHED_TV_PROGRAMMES, TIMEPASS, TYPE_OF_JOB, WEEKLY_WORKING_HOURS, 
#                  FAVE1, FAVE2, FAVE3, FAVE4, FAVE5, FAVE6, FAVE7, FAVE8, FAVE9, FAVE10, UNFAVE1, UNFAVE2, UNFAVE3, UNFAVE4, UNFAVE5, 
#                  UNFAVE6, RATING]

AD_FACE_COLS = [AD_NUM_FACES]
AD_LABEL_COLS = [AD_LABEL_FEATURE_1,AD_LABEL_FEATURE_2,AD_LABEL_FEATURE_3,AD_LABEL_FEATURE_4,AD_LABEL_FEATURE_5,
                AD_LABEL_FEATURE_6,AD_LABEL_FEATURE_7,AD_LABEL_FEATURE_8,AD_LABEL_FEATURE_9,AD_LABEL_FEATURE_10,
                AD_LABEL_FEATURE_11,AD_LABEL_FEATURE_12,AD_LABEL_FEATURE_13,AD_LABEL_FEATURE_14,AD_LABEL_FEATURE_15,
                AD_LABEL_FEATURE_16,AD_LABEL_FEATURE_17,AD_LABEL_FEATURE_18,AD_LABEL_FEATURE_19,AD_LABEL_FEATURE_20,
                AD_LABEL_FEATURE_21,AD_LABEL_FEATURE_22,AD_LABEL_FEATURE_23,AD_LABEL_FEATURE_24,AD_LABEL_FEATURE_25]
AD_OBJECT_COLS = []
AD_SAFE_SEARCH_COLS = [AD_SAFESEARCH_FEATURE_1,AD_SAFESEARCH_FEATURE_2,AD_SAFESEARCH_FEATURE_3,AD_SAFESEARCH_FEATURE_4,
                      AD_SAFESEARCH_FEATURE_5,AD_SAFESEARCH_FEATURE_6,AD_SAFESEARCH_FEATURE_7,AD_SAFESEARCH_FEATURE_8,
                      AD_SAFESEARCH_FEATURE_9,AD_SAFESEARCH_FEATURE_10,AD_SAFESEARCH_FEATURE_11,AD_SAFESEARCH_FEATURE_12,AD_SAFESEARCH_FEATURE_13]


SELECTED_AD_COLS = AD_FACE_COLS  + AD_LABEL_COLS + AD_OBJECT_COLS + AD_SAFE_SEARCH_COLS

SELECTED_HOMECOUNTRY_COLS = [HOMECOUNTRY_CANADA, HOMECOUNTRY_CZECHREPUBLIC, HOMECOUNTRY_GREATBRITAIN,
                             HOMECOUNTRY_INDIA, HOMECOUNTRY_ITALY, HOMECOUNTRY_PHILLIPINES, HOMECOUNTRY_ROMANIA,
                             HOMECOUNTRY_SAUDIARABIA, HOMECOUNTRY_SINGAPORE, HOMECOUNTRY_SLOVENIA,
                             HOMECOUNTRY_UNITEDKINGDOM, HOMECOUNTRY_UNITEDSTATESOFAMERICA]

SELECTED_INCOME_COLS = [INCOME_0, INCOME_1, INCOME_2, INCOME_3]

SELECTED_INP_COLS = [AGE, ZIP_CODE, FAVE_SPORTS, GENDER_F, GENDER_M] +                    SELECTED_AD_COLS +                    SELECTED_HOMECOUNTRY_COLS +                    SELECTED_INCOME_COLS

SELECTED_COLS = SELECTED_INP_COLS + [TARGET_COL]

SELECTED_COLS


# In[34]:


def ad_dataset_pd():
    return pd.read_csv(users_ads_rating_csv, usecols=SELECTED_COLS, dtype=str)


# In[35]:


ad_dataset_pd().sample(10)


# ## Transform Data

# In[36]:


def dict_project(d:Dict, cols:List[str]) -> Dict:
    return {k:v for k, v in d.items() if k in cols}


# In[37]:


class IndexerForVocab:
    def __init__(self, vocab_list:List[str], oov_index:int=0):
        """
        Creates a string indexer for the vocabulary with out of vocabulary (oov) indexing
        """
        self._vocab_map = {v:i+1 for i, v in enumerate(vocab_list)}
        self._oov = oov_index
        
    def __repr__(self):
        return f"Map for {len(self)} keys with 1 OOV key"
    
    def __len__(self):
        return len(self._vocab_map) + 1
        
    def index_of(self, item:str):
        """
        Index of item in the vocabulary
        """
        return self._vocab_map.get(item, self._oov)
    
    def index_of_mux(self, items:List[str]):
        return [self.index_of(i) for i in items]


# ### Age
# 
# Convert to a number and remove any outliers

# In[38]:


# Obtained from Tensorflow Data Validation APIs data-exploration/tensorflow-data-validation.ipynb

MEAN_AGE, STD_AGE, MEDIAN_AGE, MAX_AGE = 31.74, 12.07, 29, 140


# In[39]:


def fix_age(age_str:tf.string, default_age=MEDIAN_AGE) -> int:
    """Typecast age to an integer and update outliers with the default"""
    try:
        age = int(age_str)
        if age < 0 or age > MAX_AGE:
            raise ValueError(f"{age} is not a valid age")
    except:
        age = default_age
    normalized_age = (age - MEAN_AGE) / STD_AGE
    return normalized_age


# #### Visual Tests

# In[40]:


fix_age("50"), fix_age("50.5"), fix_age("-10"), fix_age("bad_age_10"), fix_age("300")


# ### Zip Code
# 
# Prepare zip-code column for one-hot encoding each character

# In[41]:


DEFAULT_ZIP_CODE, FIRST_K_ZIP_DIGITS = "00000", 2

zip_code_indexer = IndexerForVocab(string.digits + string.ascii_lowercase + string.ascii_uppercase)


# In[42]:


def fix_zip_code_tensor(zip_code:tf.string, n_digits, indexer) -> List[str]:
    """Extracts the the first n_digits as a list"""
    zip_digits = []
    try:
        if isinstance(zip_code, tf.Tensor):
            zip_code = zip_code.numpy()[0].decode('ascii', errors="ignore") # very ineffecient way
        zip_digits = list(zip_code.strip()[:n_digits])
    except:
        zip_digits = list(DEFAULT_ZIP_CODE[:n_digits])
    return tf.concat( [
        tf.one_hot(
            indexer.index_of(d), len(indexer)
        ) for d in zip_digits
    ], 0 )

def fix_zip_code(zip_code:str, n_digits, indexer) -> List[str]:
    """Extracts the the first n_digits as a list"""
    zip_digits = []
    try:
        zip_digits = list(zip_code.strip()[:n_digits])
    except:
        zip_digits = list(DEFAULT_ZIP_CODE[:n_digits])
    return np.ravel(np.eye(len(indexer))[indexer.index_of_mux(zip_digits)])


# #### Visual Tests

# In[43]:


test_zip_code_indexer = IndexerForVocab(string.digits)

(fix_zip_code("43556", 10, test_zip_code_indexer),
fix_zip_code("43556", 2, test_zip_code_indexer),
fix_zip_code("43556", 4, test_zip_code_indexer),
fix_zip_code(None, 3, test_zip_code_indexer))


# ### Favorite Sports
# 
# Two approaches,
# 1. Consider the first `K` sports mentioned by each user and one-hot encode each separately
# 2. Multi label binarize all the sports as there are only 15 unique sports

# In[44]:


FAV_SPORTS_UNKNOWN = "UNK_SPORT"
ALL_FAV_SPORTS = ['Olympic sports', 'Winter sports', 'Nothing', 'I do not like Sports', 'Equestrian sports', 'Skating sports', 'Precision sports', 'Hunting sports', 'Motor sports', 'Team sports', 'Individual sports', 'Other', 'Water sports', 'Indoor sports', 'Endurance sports']

fav_sports_binarizer = MultiLabelBinarizer()
fav_sports_binarizer.fit([ALL_FAV_SPORTS])


# In[45]:


def fav_sports_multi_select_str_to_list(sports_str:Union[str, tf.Tensor]) -> List[str]:
    # remove commas that dont separate different user selections
    # example, commas inside paranthesis of "Individual sports (Tennis, Archery, ...)" dont make new sports
    if isinstance(sports_str, tf.Tensor):
        sports_str = sports_str.numpy()[0].decode('ascii', errors="ignore")
    else:
        sports_str = sports_str.encode("ascii", errors="ignore").decode("ascii") # remove non-ascii chars
    sports = re.sub(r"\s*\(.*,.*\)\s*", "", sports_str)
    return re.split(r"\s*,\s*", sports)

def fix_fav_sports_mlb(sports_str:str) -> List[int]:
    sports = fav_sports_multi_select_str_to_list(sports_str)
    return fav_sports_binarizer.transform([sports])[0]

def fix_fav_sports_firstk(sports_str:str, first_k:int, pad_constant:int) -> List[str]:
    sports = fav_sports_multi_select_str_to_list(sports_str)
    right_pad_width = first_k - len(sports_enc)
    result = [sports + [pad_constant] * right_pad_width][:first_k]
    return result


# #### Visual Tests

# In[46]:


(
    fix_fav_sports_mlb("Individual sports (Tennis, Archery, ...), Indoor sports, Endurance sports, Skating sports"),
    fix_fav_sports_mlb("Skating sports"),
    fix_fav_sports_mlb("Individual sports (Tennis, Archery, ...)"),
    fix_fav_sports_mlb("Indoor sports, Endurance sports, Skating sports"),
)


# ### Target

# In[47]:


RATINGS_CARDINALITY = 5 # not zero based indexing i.e. ratings range from 1 to 5


# In[48]:


def create_target_pd(rating_str:str):
    return np.eye(RATINGS_CARDINALITY, dtype=int)[int(float(rating_str)) - 1]


# ## Featurize

# In[49]:


def transform_pd_X(df:pd.DataFrame, inp_cols:List[str]):
    """Original dataframe will be modified"""
    df[AGE] = df[AGE].apply(lambda age: [fix_age(age)])
    df[ZIP_CODE] = df[ZIP_CODE].apply(lambda zc: fix_zip_code(zc, n_digits=2, indexer=zip_code_indexer))
    df[FAVE_SPORTS] = df[FAVE_SPORTS].apply(fix_fav_sports_mlb)
    df[GENDER_F] = df[GENDER_F].apply(lambda gender_f: [int(gender_f)])
    df[GENDER_M] = df[GENDER_M].apply(lambda gender_m: [int(gender_m)])
    df[AD_NUM_FACES] = df[AD_NUM_FACES].apply(lambda ad_num_faces: [int(ad_num_faces)])
    
    df[AD_LABEL_FEATURE_1] = df[AD_LABEL_FEATURE_1].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_2] = df[AD_LABEL_FEATURE_2].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_3] = df[AD_LABEL_FEATURE_3].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_4] = df[AD_LABEL_FEATURE_4].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_5] = df[AD_LABEL_FEATURE_5].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_6] = df[AD_LABEL_FEATURE_6].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_7] = df[AD_LABEL_FEATURE_7].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_8] = df[AD_LABEL_FEATURE_8].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_9] = df[AD_LABEL_FEATURE_9].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_10] = df[AD_LABEL_FEATURE_10].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_11] = df[AD_LABEL_FEATURE_11].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_12] = df[AD_LABEL_FEATURE_12].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_13] = df[AD_LABEL_FEATURE_13].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_14] = df[AD_LABEL_FEATURE_14].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_15] = df[AD_LABEL_FEATURE_15].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_16] = df[AD_LABEL_FEATURE_16].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_17] = df[AD_LABEL_FEATURE_17].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_18] = df[AD_LABEL_FEATURE_18].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_19] = df[AD_LABEL_FEATURE_19].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_20] = df[AD_LABEL_FEATURE_20].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_21] = df[AD_LABEL_FEATURE_21].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_22] = df[AD_LABEL_FEATURE_22].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_23] = df[AD_LABEL_FEATURE_23].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_24] = df[AD_LABEL_FEATURE_24].apply(lambda f: [int(f)])
    df[AD_LABEL_FEATURE_25] = df[AD_LABEL_FEATURE_25].apply(lambda f: [int(f)])
    
    df[AD_SAFESEARCH_FEATURE_1] = df[AD_SAFESEARCH_FEATURE_1].apply(lambda f: [int(f)])
    df[AD_SAFESEARCH_FEATURE_2] = df[AD_SAFESEARCH_FEATURE_2].apply(lambda f: [int(f)])
    df[AD_SAFESEARCH_FEATURE_3] = df[AD_SAFESEARCH_FEATURE_3].apply(lambda f: [int(f)])
    df[AD_SAFESEARCH_FEATURE_4] = df[AD_SAFESEARCH_FEATURE_4].apply(lambda f: [int(f)])
    df[AD_SAFESEARCH_FEATURE_5] = df[AD_SAFESEARCH_FEATURE_5].apply(lambda f: [int(f)])
    df[AD_SAFESEARCH_FEATURE_6] = df[AD_SAFESEARCH_FEATURE_6].apply(lambda f: [int(f)])
    df[AD_SAFESEARCH_FEATURE_7] = df[AD_SAFESEARCH_FEATURE_7].apply(lambda f: [int(f)])
    df[AD_SAFESEARCH_FEATURE_8] = df[AD_SAFESEARCH_FEATURE_8].apply(lambda f: [int(f)])
    df[AD_SAFESEARCH_FEATURE_9] = df[AD_SAFESEARCH_FEATURE_9].apply(lambda f: [int(f)])
    df[AD_SAFESEARCH_FEATURE_10] = df[AD_SAFESEARCH_FEATURE_10].apply(lambda f: [int(f)])
    df[AD_SAFESEARCH_FEATURE_11] = df[AD_SAFESEARCH_FEATURE_11].apply(lambda f: [int(f)])
    df[AD_SAFESEARCH_FEATURE_12] = df[AD_SAFESEARCH_FEATURE_12].apply(lambda f: [int(f)])
    df[AD_SAFESEARCH_FEATURE_13] = df[AD_SAFESEARCH_FEATURE_13].apply(lambda f: [int(f)])
            
    df[HOMECOUNTRY_CANADA] = df[HOMECOUNTRY_CANADA].apply(lambda f: [int(f)])
    df[HOMECOUNTRY_CZECHREPUBLIC] = df[HOMECOUNTRY_CZECHREPUBLIC].apply(lambda f: [int(f)])
    df[HOMECOUNTRY_GREATBRITAIN] = df[HOMECOUNTRY_GREATBRITAIN].apply(lambda f: [int(f)])
    df[HOMECOUNTRY_INDIA] = df[HOMECOUNTRY_INDIA].apply(lambda f: [int(f)])
    df[HOMECOUNTRY_ITALY] = df[HOMECOUNTRY_ITALY].apply(lambda f: [int(f)])
    df[HOMECOUNTRY_PHILLIPINES] = df[HOMECOUNTRY_PHILLIPINES].apply(lambda f: [int(f)])
    df[HOMECOUNTRY_ROMANIA] = df[HOMECOUNTRY_ROMANIA].apply(lambda f: [int(f)])
    df[HOMECOUNTRY_SAUDIARABIA] = df[HOMECOUNTRY_SAUDIARABIA].apply(lambda f: [int(f)])
    df[HOMECOUNTRY_SINGAPORE] = df[HOMECOUNTRY_SINGAPORE].apply(lambda f: [int(f)])
    df[HOMECOUNTRY_SLOVENIA] = df[HOMECOUNTRY_SLOVENIA].apply(lambda f: [int(f)])
    df[HOMECOUNTRY_UNITEDKINGDOM] = df[HOMECOUNTRY_UNITEDKINGDOM].apply(lambda f: [int(f)])
    df[HOMECOUNTRY_UNITEDSTATESOFAMERICA] = df[HOMECOUNTRY_UNITEDSTATESOFAMERICA].apply(lambda f: [int(f)])

    df[INCOME_0] = df[INCOME_0].apply(lambda f: [int(f)])
    df[INCOME_1] = df[INCOME_1].apply(lambda f: [int(f)])
    df[INCOME_2] = df[INCOME_2].apply(lambda f: [int(f)])
    df[INCOME_3] = df[INCOME_3].apply(lambda f: [int(f)])

    df["X"] = df[inp_cols].apply(np.concatenate, axis=1)
    # TODO: vectorize, else inefficient to sequentially loop over all example
    X = np.array([x for x in df["X"]])
    return X


# In[50]:


def transform_pd_y(df:pd.DataFrame, target_col:str):
    """Original dataframe will be modified"""
    df["y"] = df[target_col].apply(create_target_pd)
    # TODO: vectorize, else inefficient to sequentially loop over all example
    y = np.array([y for y in df["y"]])
    return y


# In[51]:


def create_dataset_pd(inp_cols:List[str]=SELECTED_INP_COLS, target_col:str=TARGET_COL, fraction:float=1) -> pd.DataFrame:
    """Prepare the dataset for training on a fraction of all input data"""
    df = ad_dataset_pd().sample(frac=fraction)
    return transform_pd_X(df, inp_cols), transform_pd_y(df, target_col)


# ## Tensorboard
# 
# Monitor training and other stats

# In[23]:


from tensorboard import notebook


# In[24]:


get_ipython().magic('reload_ext tensorboard')


# Start tensorboard

# In[25]:


get_ipython().magic('tensorboard --logdir logs --port 6006')


# In[26]:


notebook.list()


# ## Model
# 
# Create a model and train using high level APIs like `tf.keras` and `tf.estimator`

# In[52]:


get_ipython().run_cell_magic('time', '', '\n# train_dataset = input_fn_train(BATCH_SIZE)\nX, y = create_dataset_pd()')


# In[53]:


# tf.keras.metrics.SensitivityAtSpecificity(name="ss")  # For false positive rate

keras_model_metrics = [
    "accuracy",
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'), 
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc')
]
train_histories = []


# In[54]:


# DON'T CHANGE THE EPOCHS VALUE
BATCH_SIZE = 4096
EPOCHS = 1000


# In[55]:


logdir = Path("logs")/datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    logdir, 
    histogram_freq=max(1, ceil(EPOCHS / 20)), # to control the amount of logging
#     embeddings_freq=epochs,
)
print(f"Logging tensorboard data at {logdir}")


# In[56]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, input_shape=(X.shape[1],), activation=tf.keras.layers.LeakyReLU()),
    tf.keras.layers.Dense(RATINGS_CARDINALITY , activation='softmax')
])

model.compile(
    optimizer=tf.optimizers.Adam(
        learning_rate=0.003,
        clipvalue=0.5
    ), 
#     optimizer=tf.keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True), 
#     optimizer=tf.keras.optimizers.RMSprop(lr),
#     loss=tf.nn.softmax_cross_entropy_with_logits,
    loss="categorical_crossentropy",
    metrics=keras_model_metrics
)

model.summary()


# In[57]:


get_ipython().run_cell_magic('time', '', '\ntrain_histories.append(model.fit(\n    X, y,\n    BATCH_SIZE,\n    epochs=EPOCHS, \n    callbacks=[tensorboard_callback, tfdocs.modeling.EpochDots()],\n    validation_split=0.2,\n    verbose=0\n))')


# In[58]:


metrics_df = pd.DataFrame(train_histories[-1].history) # pick the latest training history

metrics_df.tail(1) # pick the last epoch's metrics


# `Tip:` You can copy the final metrics row from above and paste it using `Shift + Cmd + V` in our [sheet](https://docs.google.com/spreadsheets/d/1v-nYiDA3elM1UP9stkB42MK0bTbuLxYJE7qAYDP8FHw/edit#gid=925421130) to accurately place all values in the respective columns
# 
# **IMPORTANT**: Please don't forget to update git version ID column after you check-in.

# ### Model Metrics with p-value

# TODO

# ## Export
# 
# Save the model for future reference

# In[30]:


model.save((logdir/"keras_saved_model").as_posix(), save_format="tf")


# ## Predict

# In[35]:


PredictionReport = namedtuple("PredictionReport", "probabilities predicted_rating confidence")

test_df = pd.DataFrame({
    AGE: ["45"],
    ZIP_CODE: ["94086"],
    FAVE_SPORTS: ["I do not like Sports"]
})

probabilities = model.predict(transform_pd_X(test_df, SELECTED_INP_COLS))
predicted_rating, confidence = np.argmax(probabilities), np.max(probabilities)

PredictionReport(probabilities, predicted_rating, confidence)


# ## Rough

# ### Featurize using Feature Columns
# 
# Create feature columns like one-hot, embeddings, bucketing from raw features created earlier

# In[ ]:


EXAMPLE_BATCH = next(iter(input_fn_train(3)))[0]


# In[ ]:


EXAMPLE_BATCH


# In[ ]:


def test_feature_column(feature_column):
    feature_layer = tf.keras.layers.DenseFeatures(feature_column)
    return feature_layer(EXAMPLE_BATCH).numpy()


# In[ ]:


age_fc = tf.feature_column.numeric_column(AGE, normalizer_fn=lambda x: (x - MEAN_AGE) / STD_AGE)


# In[ ]:


zip_fcs = [
    tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            f"{ZIP_CODE}{i}", vocabulary_list=list(string.digits), 
            num_oov_buckets=1)
    )
    for i in range(FIRST_K_ZIP_DIGITS)
]


# In[ ]:


EXAMPLE_BATCH[AGE], test_feature_column(age_fc)


# In[ ]:


{k: v for k, v in EXAMPLE_BATCH.items() if k.startswith(ZIP_CODE)}, test_feature_column(zip_fcs)


# In[ ]:


tf.keras.layers.concatenate(age_fc, zip_fcs[0])

