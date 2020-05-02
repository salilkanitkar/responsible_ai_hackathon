#!/usr/bin/env python
# coding: utf-8

# # Neural Network Model
# 
# The aim of the notebook is demo end to end pipeline for Ads prediction in Tensorflow

# In[1]:


# ! ./setup.sh # uncomment if you wish to install any new packages


# In[1]:


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
from typing import Dict, Any, Union, List, Tuple
from functools import partial
import re
import string
from sklearn.preprocessing import MultiLabelBinarizer
from math import ceil
from collections import namedtuple
from sklearn.model_selection import train_test_split


pd.set_option('display.max_rows', None)

print(f"Using Tensorflow, {tf.__version__} on Python interpreter, {sys.version_info}")


# In[2]:


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

# In[3]:


DATA_FOLDER = Path("../../dataset/")
BATCH_SIZE = 4096 # bigger the batch, faster the training but bigger the RAM needed
TARGET_COL = "Rating"

# data files path are relative DATA_FOLDER
users_ads_rating_csv = DATA_FOLDER/"users-ads-without-gcp-ratings_OHE_MLB.csv"


# In[4]:


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
# Mostlistenedmusics = 22 Columns
MOSTLISTENEDMUSICS_1 = "AlternativeMusic"
MOSTLISTENEDMUSICS_2 = "AsianPopJPoporKpop"
MOSTLISTENEDMUSICS_3 = "Blues"
MOSTLISTENEDMUSICS_4 = "ClassicalMusic"
MOSTLISTENEDMUSICS_5 = "CountryMusic"
MOSTLISTENEDMUSICS_6 = "DanceMusic"
MOSTLISTENEDMUSICS_7 = "EasyListening"
MOSTLISTENEDMUSICS_8 = "ElectronicMusic"
MOSTLISTENEDMUSICS_9 = "EuropeanMusicFolkPop"
MOSTLISTENEDMUSICS_10 = "HipHopRap"
MOSTLISTENEDMUSICS_11 = "IndiePop"
MOSTLISTENEDMUSICS_12 = "InspirationalinclGospel"
MOSTLISTENEDMUSICS_13 = "Jazz"
MOSTLISTENEDMUSICS_14 = "LatinMusic"
MOSTLISTENEDMUSICS_15 = "NewAge"
MOSTLISTENEDMUSICS_16 = "Opera"
MOSTLISTENEDMUSICS_17 = "PopPopularmusic"
MOSTLISTENEDMUSICS_18 = "RampBSoul"
MOSTLISTENEDMUSICS_19 = "Reggae"
MOSTLISTENEDMUSICS_20 = "Rock"
MOSTLISTENEDMUSICS_21 = "SingerSongwriterincFolk"
MOSTLISTENEDMUSICS_22 = "WorldMusicBeats"
# Mostreadbooks = 31 Columns
MOSTREADBOOKS_1 = "ActionandAdventure"
MOSTREADBOOKS_2 = "Anthologies"
MOSTREADBOOKS_3 = "Art"
MOSTREADBOOKS_4 = "Autobiographies"
MOSTREADBOOKS_5 = "Biographies"
MOSTREADBOOKS_6 = "Childrens"
MOSTREADBOOKS_7 = "Childrensliterature"
MOSTREADBOOKS_8 = "Comics"
MOSTREADBOOKS_9 = "Cookbooks"
MOSTREADBOOKS_10 = "Diaries"
MOSTREADBOOKS_11 = "Drama"
MOSTREADBOOKS_12 = "Encyclopedias"
MOSTREADBOOKS_13 = "Eroticfiction"
MOSTREADBOOKS_14 = "Fantasy"
MOSTREADBOOKS_15 = "Guide"
MOSTREADBOOKS_16 = "History"
MOSTREADBOOKS_17 = "Horror"
MOSTREADBOOKS_18 = "Journals"
MOSTREADBOOKS_19 = "Math"
MOSTREADBOOKS_20 = "Mystery"
MOSTREADBOOKS_21 = "Poetry"
MOSTREADBOOKS_22 = "Prayerbooks"
MOSTREADBOOKS_23 = "Religious"
MOSTREADBOOKS_24 = "Romance"
MOSTREADBOOKS_25 = "Satire"
MOSTREADBOOKS_26 = "Science"
MOSTREADBOOKS_27 = "Sciencefiction"
MOSTREADBOOKS_28 = "Selfhelp"
MOSTREADBOOKS_29 = "Series"
MOSTREADBOOKS_30 = "Travel"
MOSTREADBOOKS_31 = "Trilogies"
# Mostwatchedmovies = 21 Columns
MOSTWATCHEDMOVIES_1 = "Mostwatchedmovies_Action"
MOSTWATCHEDMOVIES_2 = "Mostwatchedmovies_Adventure"
MOSTWATCHEDMOVIES_3 = "Mostwatchedmovies_Animation"
MOSTWATCHEDMOVIES_4 = "Mostwatchedmovies_Biography"
MOSTWATCHEDMOVIES_5 = "Mostwatchedmovies_Comedy"
MOSTWATCHEDMOVIES_6 = "Mostwatchedmovies_CrimeandGangster"
MOSTWATCHEDMOVIES_7 = "Mostwatchedmovies_Documentary"
MOSTWATCHEDMOVIES_8 = "Mostwatchedmovies_Drama"
MOSTWATCHEDMOVIES_9 = "Mostwatchedmovies_EpicHistorical"
MOSTWATCHEDMOVIES_10 = "Mostwatchedmovies_Erotic"
MOSTWATCHEDMOVIES_11 = "Mostwatchedmovies_Family"
MOSTWATCHEDMOVIES_12 = "Mostwatchedmovies_Fantasy"
MOSTWATCHEDMOVIES_13 = "Mostwatchedmovies_Horror"
MOSTWATCHEDMOVIES_14 = "Mostwatchedmovies_Musical"
MOSTWATCHEDMOVIES_15 = "Mostwatchedmovies_Mystery"
MOSTWATCHEDMOVIES_16 = "Mostwatchedmovies_Romance"
MOSTWATCHEDMOVIES_17 = "Mostwatchedmovies_SciFi"
MOSTWATCHEDMOVIES_18 = "Mostwatchedmovies_Sport"
MOSTWATCHEDMOVIES_19 = "Mostwatchedmovies_Thriller"
MOSTWATCHEDMOVIES_20 = "Mostwatchedmovies_War"
MOSTWATCHEDMOVIES_21 = "Mostwatchedmovies_Western"

RATING = "Rating"
AD_NUM_FACES = "ad_num_faces"
AD_LABEL_FEATURE_1 = 'ad_isAdvertising'
AD_LABEL_FEATURE_2 = 'ad_isBrand'
AD_LABEL_FEATURE_3 = 'ad_isElectronicdevice'
AD_LABEL_FEATURE_4 = 'ad_isElectronics'
AD_LABEL_FEATURE_5 = 'ad_isFashionaccessory'
AD_LABEL_FEATURE_6 = 'ad_isFictionalcharacter'
AD_LABEL_FEATURE_7 = 'ad_isFont'
AD_LABEL_FEATURE_8 = 'ad_isFurniture'
AD_LABEL_FEATURE_9 = 'ad_isGadget'
AD_LABEL_FEATURE_10 = 'ad_isGames'
AD_LABEL_FEATURE_11 = 'ad_isGraphicdesign'
AD_LABEL_FEATURE_12 = 'ad_isGraphics'
AD_LABEL_FEATURE_13 = 'ad_isJewellery'
AD_LABEL_FEATURE_14 = 'ad_isLine'
AD_LABEL_FEATURE_15 = 'ad_isLogo'
AD_LABEL_FEATURE_16 = 'ad_isMagenta'
AD_LABEL_FEATURE_17 = 'ad_isMaterialproperty'
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
AD_OBJECT_FEATURE_1 = 'ad_isAnimal'
AD_OBJECT_FEATURE_2 ='ad_isBelt'
AD_OBJECT_FEATURE_3 ='ad_isBottle'
AD_OBJECT_FEATURE_4 ='ad_isBox'
AD_OBJECT_FEATURE_5 ='ad_isCameralens'
AD_OBJECT_FEATURE_6 ='ad_isChair'
AD_OBJECT_FEATURE_7 ='ad_isClothing'
AD_OBJECT_FEATURE_8 ='ad_isEarrings'
AD_OBJECT_FEATURE_9 ='ad_isFood'
AD_OBJECT_FEATURE_10 ='ad_isHat'
AD_OBJECT_FEATURE_11 ='ad_isLuggagebags'
AD_OBJECT_FEATURE_12 ='ad_isMobilephone'
AD_OBJECT_FEATURE_13 ='ad_isNecklace'
AD_OBJECT_FEATURE_14 ='ad_isPackagedgoods'
AD_OBJECT_FEATURE_15 ='ad_isPants'
AD_OBJECT_FEATURE_16 ='ad_isPen'
AD_OBJECT_FEATURE_17 ='ad_isPerson'
AD_OBJECT_FEATURE_18 ='ad_isPillow'
AD_OBJECT_FEATURE_19 ='ad_isPoster'
AD_OBJECT_FEATURE_20 ='ad_isShoe'
AD_OBJECT_FEATURE_21 ='ad_isTop'
AD_OBJECT_FEATURE_22 ='ad_isToy'
AD_OBJECT_FEATURE_23 ='ad_isWatch'
AD_OBJECT_FEATURE_24  ='ad_isWheel'



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
    MOSTLISTENEDMUSICS_1: "**",
    MOSTLISTENEDMUSICS_2: "**",
    MOSTLISTENEDMUSICS_3: "**",
    MOSTLISTENEDMUSICS_4: "**",
    MOSTLISTENEDMUSICS_5: "**",
    MOSTLISTENEDMUSICS_6: "**",
    MOSTLISTENEDMUSICS_7: "**",
    MOSTLISTENEDMUSICS_8: "**",
    MOSTLISTENEDMUSICS_9: "**",
    MOSTLISTENEDMUSICS_10: "**",
    MOSTLISTENEDMUSICS_11: "**",
    MOSTLISTENEDMUSICS_12: "**",
    MOSTLISTENEDMUSICS_13: "**",
    MOSTLISTENEDMUSICS_14: "**",
    MOSTLISTENEDMUSICS_15: "**",
    MOSTLISTENEDMUSICS_16: "**",
    MOSTLISTENEDMUSICS_17: "**",
    MOSTLISTENEDMUSICS_18: "**",
    MOSTLISTENEDMUSICS_19: "**",
    MOSTLISTENEDMUSICS_20: "**",
    MOSTLISTENEDMUSICS_21: "**",
    MOSTLISTENEDMUSICS_22: "**",
    MOSTREADBOOKS_1: "**",
    MOSTREADBOOKS_2: "**",
    MOSTREADBOOKS_3: "**",
    MOSTREADBOOKS_4: "**",
    MOSTREADBOOKS_5: "**",
    MOSTREADBOOKS_6: "**",
    MOSTREADBOOKS_7: "**",
    MOSTREADBOOKS_8: "**",
    MOSTREADBOOKS_9: "**",
    MOSTREADBOOKS_10: "**",
    MOSTREADBOOKS_11: "**",
    MOSTREADBOOKS_12: "**",
    MOSTREADBOOKS_13: "**",
    MOSTREADBOOKS_14: "**",
    MOSTREADBOOKS_15: "**",
    MOSTREADBOOKS_16: "**",
    MOSTREADBOOKS_17: "**",
    MOSTREADBOOKS_18: "**",
    MOSTREADBOOKS_19: "**",
    MOSTREADBOOKS_20: "**",
    MOSTREADBOOKS_21: "**",
    MOSTREADBOOKS_22: "**",
    MOSTREADBOOKS_23: "**",
    MOSTREADBOOKS_24: "**",
    MOSTREADBOOKS_25: "**",
    MOSTREADBOOKS_26: "**",
    MOSTREADBOOKS_27: "**",
    MOSTREADBOOKS_28: "**",
    MOSTREADBOOKS_29: "**",
    MOSTREADBOOKS_30: "**",
    MOSTREADBOOKS_31: "**",
    MOSTWATCHEDMOVIES_1: "**",
    MOSTWATCHEDMOVIES_2: "**",
    MOSTWATCHEDMOVIES_3: "**",
    MOSTWATCHEDMOVIES_4: "**",
    MOSTWATCHEDMOVIES_5: "**",
    MOSTWATCHEDMOVIES_6: "**",
    MOSTWATCHEDMOVIES_7: "**",
    MOSTWATCHEDMOVIES_8: "**",
    MOSTWATCHEDMOVIES_9: "**",
    MOSTWATCHEDMOVIES_10: "**",
    MOSTWATCHEDMOVIES_11: "**",
    MOSTWATCHEDMOVIES_12: "**",
    MOSTWATCHEDMOVIES_13: "**",
    MOSTWATCHEDMOVIES_14: "**",
    MOSTWATCHEDMOVIES_15: "**",
    MOSTWATCHEDMOVIES_16: "**",
    MOSTWATCHEDMOVIES_17: "**",
    MOSTWATCHEDMOVIES_18: "**",
    MOSTWATCHEDMOVIES_19: "**",
    MOSTWATCHEDMOVIES_20: "**",
    MOSTWATCHEDMOVIES_21: "**",    
    RATING: "**",
    AD_NUM_FACES: "**"
}


AD_FACE_COLS = [AD_NUM_FACES]
AD_LABEL_COLS = [AD_LABEL_FEATURE_1,AD_LABEL_FEATURE_2,AD_LABEL_FEATURE_3,AD_LABEL_FEATURE_4,AD_LABEL_FEATURE_5,
                AD_LABEL_FEATURE_6,AD_LABEL_FEATURE_7,AD_LABEL_FEATURE_8,AD_LABEL_FEATURE_9,AD_LABEL_FEATURE_10,
                AD_LABEL_FEATURE_11,AD_LABEL_FEATURE_12,AD_LABEL_FEATURE_13,AD_LABEL_FEATURE_14,AD_LABEL_FEATURE_15,
                AD_LABEL_FEATURE_16,AD_LABEL_FEATURE_17,AD_LABEL_FEATURE_18,AD_LABEL_FEATURE_19,AD_LABEL_FEATURE_20,
                AD_LABEL_FEATURE_21,AD_LABEL_FEATURE_22,AD_LABEL_FEATURE_23,AD_LABEL_FEATURE_24,AD_LABEL_FEATURE_25]

AD_OBJECT_COLS = [AD_OBJECT_FEATURE_1,AD_OBJECT_FEATURE_2,AD_OBJECT_FEATURE_3,AD_OBJECT_FEATURE_4,AD_OBJECT_FEATURE_5,
                AD_OBJECT_FEATURE_6,AD_OBJECT_FEATURE_7,AD_OBJECT_FEATURE_8,AD_OBJECT_FEATURE_9,AD_OBJECT_FEATURE_10,
                AD_OBJECT_FEATURE_11,AD_OBJECT_FEATURE_12,AD_OBJECT_FEATURE_13,AD_OBJECT_FEATURE_14,AD_OBJECT_FEATURE_15,
                AD_OBJECT_FEATURE_16,AD_OBJECT_FEATURE_17,AD_OBJECT_FEATURE_18,AD_OBJECT_FEATURE_19,AD_OBJECT_FEATURE_20,
                AD_OBJECT_FEATURE_21,AD_OBJECT_FEATURE_22,AD_OBJECT_FEATURE_23,AD_OBJECT_FEATURE_24]


AD_SAFE_SEARCH_COLS = [AD_SAFESEARCH_FEATURE_1,AD_SAFESEARCH_FEATURE_2,AD_SAFESEARCH_FEATURE_3,AD_SAFESEARCH_FEATURE_4,
                      AD_SAFESEARCH_FEATURE_5,AD_SAFESEARCH_FEATURE_6,AD_SAFESEARCH_FEATURE_7,AD_SAFESEARCH_FEATURE_8,
                      AD_SAFESEARCH_FEATURE_9,AD_SAFESEARCH_FEATURE_10,AD_SAFESEARCH_FEATURE_11,AD_SAFESEARCH_FEATURE_12,AD_SAFESEARCH_FEATURE_13]


SELECTED_AD_COLS = AD_FACE_COLS  + AD_LABEL_COLS + AD_OBJECT_COLS + AD_SAFE_SEARCH_COLS

SELECTED_HOMECOUNTRY_COLS = [HOMECOUNTRY_CANADA, HOMECOUNTRY_CZECHREPUBLIC, HOMECOUNTRY_GREATBRITAIN,
                             HOMECOUNTRY_INDIA, HOMECOUNTRY_ITALY, HOMECOUNTRY_PHILLIPINES, HOMECOUNTRY_ROMANIA,
                             HOMECOUNTRY_SAUDIARABIA, HOMECOUNTRY_SINGAPORE, HOMECOUNTRY_SLOVENIA,
                             HOMECOUNTRY_UNITEDKINGDOM, HOMECOUNTRY_UNITEDSTATESOFAMERICA]

SELECTED_INCOME_COLS = [INCOME_0, INCOME_1, INCOME_2, INCOME_3]

SELECTED_MOSTLISTENEDMUSICS_COLS = [MOSTLISTENEDMUSICS_1, MOSTLISTENEDMUSICS_2, MOSTLISTENEDMUSICS_3,
                                    MOSTLISTENEDMUSICS_4, MOSTLISTENEDMUSICS_5, MOSTLISTENEDMUSICS_6,
                                    MOSTLISTENEDMUSICS_7, MOSTLISTENEDMUSICS_8, MOSTLISTENEDMUSICS_9,
                                    MOSTLISTENEDMUSICS_10, MOSTLISTENEDMUSICS_11, MOSTLISTENEDMUSICS_12,
                                    MOSTLISTENEDMUSICS_13, MOSTLISTENEDMUSICS_14, MOSTLISTENEDMUSICS_15,
                                    MOSTLISTENEDMUSICS_16, MOSTLISTENEDMUSICS_17, MOSTLISTENEDMUSICS_18,
                                    MOSTLISTENEDMUSICS_19, MOSTLISTENEDMUSICS_20, MOSTLISTENEDMUSICS_21,
                                    MOSTLISTENEDMUSICS_22]

SELECTED_MOSTREADBOOKS_COLS = [MOSTREADBOOKS_1, MOSTREADBOOKS_2, MOSTREADBOOKS_3, MOSTREADBOOKS_4,
                               MOSTREADBOOKS_5, MOSTREADBOOKS_6, MOSTREADBOOKS_7, MOSTREADBOOKS_8,
                               MOSTREADBOOKS_9, MOSTREADBOOKS_10, MOSTREADBOOKS_11, MOSTREADBOOKS_12,
                               MOSTREADBOOKS_13, MOSTREADBOOKS_14, MOSTREADBOOKS_15, MOSTREADBOOKS_16,
                               MOSTREADBOOKS_17, MOSTREADBOOKS_18, MOSTREADBOOKS_19, MOSTREADBOOKS_20,
                               MOSTREADBOOKS_21, MOSTREADBOOKS_22, MOSTREADBOOKS_23, MOSTREADBOOKS_24,
                               MOSTREADBOOKS_25, MOSTREADBOOKS_26, MOSTREADBOOKS_27, MOSTREADBOOKS_28,
                               MOSTREADBOOKS_29, MOSTREADBOOKS_30, MOSTREADBOOKS_31] 

SELECTED_MOSTWATCHEDMOVIES_COLS = [MOSTWATCHEDMOVIES_1, MOSTWATCHEDMOVIES_2, MOSTWATCHEDMOVIES_3,
                                   MOSTWATCHEDMOVIES_4, MOSTWATCHEDMOVIES_5, MOSTWATCHEDMOVIES_6,
                                   MOSTWATCHEDMOVIES_7, MOSTWATCHEDMOVIES_8, MOSTWATCHEDMOVIES_9,
                                   MOSTWATCHEDMOVIES_10, MOSTWATCHEDMOVIES_11, MOSTWATCHEDMOVIES_12,
                                   MOSTWATCHEDMOVIES_13, MOSTWATCHEDMOVIES_14, MOSTWATCHEDMOVIES_15,
                                   MOSTWATCHEDMOVIES_16, MOSTWATCHEDMOVIES_17, MOSTWATCHEDMOVIES_18,
                                   MOSTWATCHEDMOVIES_19, MOSTWATCHEDMOVIES_20, MOSTWATCHEDMOVIES_21]
                               
SELECTED_INP_COLS = [AGE, ZIP_CODE, FAVE_SPORTS, GENDER_F, GENDER_M] +                    SELECTED_AD_COLS +                    SELECTED_HOMECOUNTRY_COLS +                    SELECTED_INCOME_COLS +                    SELECTED_MOSTLISTENEDMUSICS_COLS +                    SELECTED_MOSTREADBOOKS_COLS +                    SELECTED_MOSTWATCHEDMOVIES_COLS

SELECTED_COLS = SELECTED_INP_COLS + [TARGET_COL]

print(SELECTED_COLS)


# In[5]:


def ad_dataset_pd():
    return pd.read_csv(users_ads_rating_csv, usecols=SELECTED_COLS, dtype=str)


# In[6]:


ad_dataset_pd().sample(5).T


# ## Transform Data

# In[7]:


def dict_project(d:Dict, cols:List[str]) -> Dict:
    return {k:v for k, v in d.items() if k in cols}


# In[8]:


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

# In[9]:


# Obtained from Tensorflow Data Validation APIs data-exploration/tensorflow-data-validation.ipynb

MEAN_AGE, STD_AGE, MEDIAN_AGE, MAX_AGE = 31.74, 12.07, 29, 140


# In[10]:


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

# In[11]:


fix_age("50"), fix_age("50.5"), fix_age("-10"), fix_age("bad_age_10"), fix_age("300")


# ### Zip Code
# 
# Prepare zip-code column for one-hot encoding each character

# In[12]:


DEFAULT_ZIP_CODE, FIRST_K_ZIP_DIGITS = "00000", 2

zip_code_indexer = IndexerForVocab(string.digits + string.ascii_lowercase + string.ascii_uppercase)


# In[13]:


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

# In[14]:


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

# In[15]:


FAV_SPORTS_UNKNOWN = "UNK_SPORT"
ALL_FAV_SPORTS = ['Olympic sports', 'Winter sports', 'Nothing', 'I do not like Sports', 'Equestrian sports', 'Skating sports', 'Precision sports', 'Hunting sports', 'Motor sports', 'Team sports', 'Individual sports', 'Other', 'Water sports', 'Indoor sports', 'Endurance sports']

fav_sports_binarizer = MultiLabelBinarizer()
fav_sports_binarizer.fit([ALL_FAV_SPORTS])


# In[16]:


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

# In[17]:


(
    fix_fav_sports_mlb("Individual sports (Tennis, Archery, ...), Indoor sports, Endurance sports, Skating sports"),
    fix_fav_sports_mlb("Skating sports"),
    fix_fav_sports_mlb("Individual sports (Tennis, Archery, ...)"),
    fix_fav_sports_mlb("Indoor sports, Endurance sports, Skating sports"),
)


# ### Target

# In[18]:


RATINGS_CARDINALITY = 5 # not zero based indexing i.e. ratings range from 1 to 5


# In[19]:


def create_target_pd(rating_str:str):
    return np.eye(RATINGS_CARDINALITY, dtype=int)[int(float(rating_str)) - 1]


# ## Featurize

# In[20]:


def transform_pd_X(df:pd.DataFrame, inp_cols:List[str]):
    """Original dataframe will be modified"""
    df[AGE] = df[AGE].apply(lambda age: [fix_age(age)])
    df[ZIP_CODE] = df[ZIP_CODE].apply(lambda zc: fix_zip_code(zc, n_digits=2, indexer=zip_code_indexer))
    df[FAVE_SPORTS] = df[FAVE_SPORTS].apply(fix_fav_sports_mlb)

    int_cols = [GENDER_F, GENDER_M, AD_NUM_FACES] +               AD_LABEL_COLS +               AD_SAFE_SEARCH_COLS +               SELECTED_HOMECOUNTRY_COLS +               SELECTED_INCOME_COLS +               SELECTED_MOSTLISTENEDMUSICS_COLS +               SELECTED_MOSTREADBOOKS_COLS +               SELECTED_MOSTWATCHEDMOVIES_COLS +               AD_OBJECT_COLS
    
    df[int_cols] = df[int_cols].applymap(lambda f: [int(f)])

    df["X"] = df[inp_cols].apply(np.concatenate, axis=1)
    # TODO: vectorize, else inefficient to sequentially loop over all examples
    X = np.array([x for x in df["X"]])
    return X


# In[21]:


def transform_pd_y(df:pd.DataFrame, target_col:str):
    """Original dataframe will be modified"""
    df["y"] = df[target_col].apply(create_target_pd)
    # TODO: vectorize, else inefficient to sequentially loop over all examples
    y = np.array([y for y in df["y"]])
    return y


# In[22]:


def create_dataset_pd(inp_cols:List[str]=SELECTED_INP_COLS, target_col:str=TARGET_COL, fraction:float=1, test_frac:float=0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare the dataset for training on a fraction of all input data"""
    df = ad_dataset_pd().sample(frac=fraction)
    X, y = transform_pd_X(df, inp_cols), transform_pd_y(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=RANDOM_SEED)
    return X_train, X_test, y_train, y_test


# ## Tensorboard
# 
# Monitor training and other stats

# In[26]:


from tensorboard import notebook


# In[27]:


get_ipython().magic('reload_ext tensorboard')


# Start tensorboard

# In[28]:


get_ipython().magic('tensorboard --logdir logs --port 6006')


# In[29]:


notebook.list()


# ## Model
# 
# Create a model and train using high level APIs like `tf.keras` and `tf.estimator`

# In[23]:


get_ipython().run_cell_magic('time', '', '\nX_train, X_test, y_train, y_test = create_dataset_pd()')


# In[24]:


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


# In[25]:


# DON'T CHANGE THE EPOCHS VALUE
BATCH_SIZE = 4096
EPOCHS = 1000


# In[26]:


logdir = Path("logs")/datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    logdir, 
    histogram_freq=max(1, ceil(EPOCHS / 20)), # to control the amount of logging
#     embeddings_freq=epochs,
)
print(f"Logging tensorboard data at {logdir}")


# In[27]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, input_shape=(X_train.shape[1],), activation=tf.keras.layers.LeakyReLU()),
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


# In[28]:


get_ipython().run_cell_magic('time', '', '\ntrain_histories.append(model.fit(\n    X_train, y_train,\n    BATCH_SIZE,\n    epochs=EPOCHS, \n    callbacks=[tensorboard_callback, tfdocs.modeling.EpochDots()],\n    validation_data=(X_test, y_test),\n    verbose=0\n))')


# In[29]:


metrics_df = pd.DataFrame(train_histories[-1].history) # pick the latest training history

metrics_df.tail(1) # pick the last epoch's metrics


# `Tip:` You can copy the final metrics row from above and paste it using `Shift + Cmd + V` in our [sheet](https://docs.google.com/spreadsheets/d/1v-nYiDA3elM1UP9stkB42MK0bTbuLxYJE7qAYDP8FHw/edit#gid=925421130) to accurately place all values in the respective columns
# 
# **IMPORTANT**: Please don't forget to update git version ID column after you check-in.

# ### Model Metrics with p-value
# 
# TODO: Run multiple times on different samples of `y_test` to compute p-value

# In[31]:


from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score, f1_score
import sklearn
from collections import OrderedDict

sklearn.__version__


# In[32]:


y_prob = model.predict(X_test, BATCH_SIZE)
y_true = y_test
y_pred = (y_prob / np.max(y_prob, axis=1).reshape(-1, 1)).astype(int) # convert probabilities to predictions


# In[33]:


pd.DataFrame(OrderedDict({
    "macro_roc_auc_ovo": [roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro")],
    "weighted_roc_auc_ovo": roc_auc_score(y_test, y_prob, multi_class="ovo", average="weighted"),
    "macro_roc_auc_ovr": roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"),
    "weighted_roc_auc_ovr": roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"),
    "weighted_precision": precision_score(y_test, y_pred, average="weighted"),
    "weighted_recall": recall_score(y_test, y_pred, average="weighted"),
    "weighted_f1": f1_score(y_test, y_pred, average="weighted")
}))


# Also paste the above numbers to our [sheet](https://docs.google.com/spreadsheets/d/1v-nYiDA3elM1UP9stkB42MK0bTbuLxYJE7qAYDP8FHw/edit#gid=925421130&range=W1:AC1)

# In[54]:


print(classification_report(y_true, y_pred))


# ## Export
# 
# Save the model for future reference

# In[30]:


model.save((logdir/"keras_saved_model").as_posix(), save_format="tf")


# ## Predict

# In[ ]:


PredictionReport = namedtuple("PredictionReport", "probabilities predicted_rating confidence")

# Create a dataframe with all SELECTED_INP_COLS
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

