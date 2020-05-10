#!/usr/bin/env python
# coding: utf-8

# ! ./setup.sh # uncomment if you wish to install any new packages


get_ipython().run_line_magic('matplotlib', 'inline')


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
from sklearn.metrics import confusion_matrix
from math import ceil
from collections import namedtuple
from sklearn.model_selection import train_test_split
import pickle

import chakin
import json
import os
from collections import defaultdict
import zipfile
import sqlite3
import logging
from tempfile import TemporaryDirectory
from fastprogress import progress_bar
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import concatenate


from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset  # To handle the data

import seaborn as sns

from EmbeddingFactory import EmbeddingFactory

pd.set_option('display.max_rows', None)

print(f"Using Tensorflow, {tf.__version__} on Python interpreter, {sys.version_info}")


RANDOM_SEED = int(time.time())

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"Using random seed, {RANDOM_SEED}")


DATA_FOLDER = Path("../../dataset/")
BATCH_SIZE = 4096 # bigger the batch, faster the training but bigger the RAM needed
TARGET_COL = "Rating"

# data files path are relative DATA_FOLDER
users_ads_rating_csv = DATA_FOLDER/"users-ads-without-gcp-ratings_OHE_MLB_FAV_UNFAV_Merged.csv"


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
# Mostwatchedtvprogrammes = 11 Columns
MOSTWATCHEDTVPROGRAMMES_1 = "Mostwatchedtvprogrammes_Childrens"
MOSTWATCHEDTVPROGRAMMES_2 = "Mostwatchedtvprogrammes_Comedy"
MOSTWATCHEDTVPROGRAMMES_3 = "Mostwatchedtvprogrammes_Drama"
MOSTWATCHEDTVPROGRAMMES_4 = "Mostwatchedtvprogrammes_EntertainmentVarietyShows"
MOSTWATCHEDTVPROGRAMMES_5 = "Mostwatchedtvprogrammes_Factual"
MOSTWATCHEDTVPROGRAMMES_6 = "Mostwatchedtvprogrammes_Learning"
MOSTWATCHEDTVPROGRAMMES_7 = "Mostwatchedtvprogrammes_Music"
MOSTWATCHEDTVPROGRAMMES_8 = "Mostwatchedtvprogrammes_News"
MOSTWATCHEDTVPROGRAMMES_9 = "Mostwatchedtvprogrammes_ReligionampEthics"
MOSTWATCHEDTVPROGRAMMES_10 = "Mostwatchedtvprogrammes_Sport"
MOSTWATCHEDTVPROGRAMMES_11 = "Mostwatchedtvprogrammes_Weather"

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
FAV = 'fav'
UNFAV = 'unfav'


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
    MOSTWATCHEDTVPROGRAMMES_1: "**",
    MOSTWATCHEDTVPROGRAMMES_2: "**",
    MOSTWATCHEDTVPROGRAMMES_3: "**",
    MOSTWATCHEDTVPROGRAMMES_4: "**",
    MOSTWATCHEDTVPROGRAMMES_5: "**",
    MOSTWATCHEDTVPROGRAMMES_6: "**",
    MOSTWATCHEDTVPROGRAMMES_7: "**",
    MOSTWATCHEDTVPROGRAMMES_8: "**",
    MOSTWATCHEDTVPROGRAMMES_9: "**",
    MOSTWATCHEDTVPROGRAMMES_10: "**",
    MOSTWATCHEDTVPROGRAMMES_11: "**",    
    RATING: "**",
    AD_NUM_FACES: "**",
    FAV: "**",
    UNFAV: "**"
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

SELECTED_MOSTWATCHEDTVPROGRAMMES_COLS = [MOSTWATCHEDTVPROGRAMMES_1, MOSTWATCHEDTVPROGRAMMES_2,
                                         MOSTWATCHEDTVPROGRAMMES_3, MOSTWATCHEDTVPROGRAMMES_4,
                                         MOSTWATCHEDTVPROGRAMMES_5, MOSTWATCHEDTVPROGRAMMES_6,
                                         MOSTWATCHEDTVPROGRAMMES_7, MOSTWATCHEDTVPROGRAMMES_8,
                                         MOSTWATCHEDTVPROGRAMMES_9, MOSTWATCHEDTVPROGRAMMES_10,
                                         MOSTWATCHEDTVPROGRAMMES_11]
                                   
SELECTED_INP_COLS = [AGE, ZIP_CODE, FAVE_SPORTS, GENDER_F, GENDER_M] +                    SELECTED_AD_COLS +                    SELECTED_HOMECOUNTRY_COLS +                    SELECTED_INCOME_COLS +                    SELECTED_MOSTLISTENEDMUSICS_COLS +                    SELECTED_MOSTREADBOOKS_COLS +                    SELECTED_MOSTWATCHEDMOVIES_COLS +                    SELECTED_MOSTWATCHEDTVPROGRAMMES_COLS

EMBED_COLS = [FAV, UNFAV]

SELECTED_COLS = SELECTED_INP_COLS + [TARGET_COL]

print(SELECTED_COLS)


def reweigh_biased_cols(bias_cols,df):
   
    X = df[bias_cols]
    y = df[[RATING]]
    
    # Assumtion for each column (except AGE) value 1 : privilaged , value 0 : non-privilaged
    priv_cols = []
    
    for col in bias_cols: 
        
        if col == AGE:
            X[AGE] = pd.to_numeric(X[AGE])
            bucket_boundaries = [0, 20, 40, 100] # refer pandas.cut() for syntax on binning
            X["Age_bucket"] = pd.cut(X[AGE], bins=bucket_boundaries, labels=["young", "middle-age", "old"])
            X['privilaged_age_bucket'] = 0.0
            #considering young and old as privilaged group
            X.loc[X['Age_bucket'] != "young", 'privilaged_age_bucket'] = 1.0 
            priv_cols.append('privilaged_age_bucket')  
            
        elif col == GENDER:
            X[col] = pd.to_numeric(X[col])
            priv_cols.append(col)  
        else:
            raise NotImplementedError(f"Don't know how to reweigh {col}")
    
    y[RATING] = pd.to_numeric(y[RATING])
    y.loc[y[RATING] != 1.0, RATING] = 2.0
    y[RATING] = y[RATING] - 1.0
    
    X = X[priv_cols]
    
    train_pp_bld = BinaryLabelDataset(df=pd.concat((X, y),axis=1),
                                      label_names=[RATING],
                                      protected_attribute_names=priv_cols,
                                      favorable_label=1.0,
                                      unfavorable_label=0.0)
    
    privileged_dict = {}
    unprivileged_dict = {}
    for col in priv_cols:
        privileged_dict[col] = 1
        unprivileged_dict[col] = 0
    
    privileged_groups = [privileged_dict]
    unprivileged_groups = [unprivileged_dict]
    
    
    # Create the metric object
    metric_train_bld = BinaryLabelDatasetMetric(train_pp_bld,
                                            unprivileged_groups=unprivileged_groups,
                                            privileged_groups=privileged_groups)
    
    
    rw = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    train_pp_bld_f = rw.fit_transform(train_pp_bld)
    
    #df["fairness_privilage_weight"] = train_pp_bld_f.instance_weights
    #df["fairness_privilage_weight"] = df["fairness_privilage_weight"].astype(str)
    
    return train_pp_bld_f.instance_weights


def ad_dataset_pd(usecols:List[str]=None, **read_csv_kwargs):
    """
    Read from csv files given set of columns into Pandas Dataframe
    """
    return pd.read_csv(users_ads_rating_csv, usecols=usecols, dtype=str, **read_csv_kwargs)


ad_dataset_pd(SELECTED_COLS).sample(5).T


chakin.search(lang='English')


WORD_VEC_DIMENSIONS = 50


get_ipython().run_cell_magic('time', '', '\nembedding_index = EmbeddingFactory(Path("./embeddings/"), "GloVe.6B.50d", WORD_VEC_DIMENSIONS, nrows=None, skiprows=None)')


def transform_embed_col(s:pd.Series, t:Tokenizer, maxlen:int=None):
    """Tokenizes each row in s using t and pads them to equal length of maxlen. Computes maxlen if not provided"""
    # integer encode the text data
    encoded_col = t.texts_to_sequences(s)

    # calculate max len of vector and make length equal by padding with zeros
    if maxlen is None:
        maxlen = max(len(x) for x in encoded_col) 
        
    return pad_sequences(encoded_col, maxlen=maxlen, padding='post'), maxlen


def create_embedding_data(df:pd.DataFrame) -> Dict:
    """Compute metadata and embedding matrix for each embedding column in df"""
    
    es = defaultdict(dict) # embedding_store
    for embed_col in EMBED_COLS:
        # Input dataframe and embed_col
        t = Tokenizer()
        t.fit_on_texts(df[embed_col])
        es[embed_col]["tokenizer"] = t

        # UNK added as tokenizer starts indexing from 1
        words = ["UNK"] + list(t.word_index.keys()) # in order of tokenizer
        es[embed_col]["vocab_size"] = len(words)
        es[embed_col]["padded_col"], es[embed_col]["maxlen"] = transform_embed_col(df[embed_col], t)

        # create a weight matrix
        embeddings = dict.fromkeys(words, " ".join(["0"] * WORD_VEC_DIMENSIONS)) # default embeddings to all words as 0
        embeddings.update(dict(embedding_index.fetch_word_vectors(words))) # update for known words
        # reorder to match tokenizer's indexing
        emb_matrix = pd.DataFrame.from_dict(embeddings, orient="index").loc[words, 0].str.split(" ", expand=True).to_numpy().astype(np.float16)
        es[embed_col]["embed_matrix"] = emb_matrix
        assert emb_matrix.shape[0] == len(words), f"For {embed_col}, not all words have embeddings"
        
    return es


create_embedding_data(ad_dataset_pd(EMBED_COLS).sample(n=3))


def dict_project(d:Dict, cols:List[str]) -> Dict:
    """Returns a new dictionary with only cols keys"""
    return {k:v for k, v in d.items() if k in cols}


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


# Obtained from Tensorflow Data Validation APIs data-exploration/tensorflow-data-validation.ipynb

MEAN_AGE, STD_AGE, MEDIAN_AGE, MAX_AGE = 31.74, 12.07, 29, 140


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


fix_age("50"), fix_age("50.5"), fix_age("-10"), fix_age("bad_age_10"), fix_age("300")


DEFAULT_ZIP_CODE, FIRST_K_ZIP_DIGITS = "00000", 2

zip_code_indexer = IndexerForVocab(string.digits + string.ascii_lowercase + string.ascii_uppercase)


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


test_zip_code_indexer = IndexerForVocab(string.digits)

(fix_zip_code("43556", 10, test_zip_code_indexer),
fix_zip_code("43556", 2, test_zip_code_indexer),
fix_zip_code("43556", 4, test_zip_code_indexer),
fix_zip_code(None, 3, test_zip_code_indexer))


FAV_SPORTS_UNKNOWN = "UNK_SPORT"
ALL_FAV_SPORTS = ['Olympic sports', 'Winter sports', 'Nothing', 'I do not like Sports', 'Equestrian sports', 'Skating sports', 'Precision sports', 'Hunting sports', 'Motor sports', 'Team sports', 'Individual sports', 'Other', 'Water sports', 'Indoor sports', 'Endurance sports']

fav_sports_binarizer = MultiLabelBinarizer()
fav_sports_binarizer.fit([ALL_FAV_SPORTS])


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


(
    fix_fav_sports_mlb("Individual sports (Tennis, Archery, ...), Indoor sports, Endurance sports, Skating sports"),
    fix_fav_sports_mlb("Skating sports"),
    fix_fav_sports_mlb("Individual sports (Tennis, Archery, ...)"),
    fix_fav_sports_mlb("Indoor sports, Endurance sports, Skating sports"),
)


RATINGS_CARDINALITY = 2 # not zero based indexing i.e. ratings range from 1 to 5


def create_target_pd(rating_str:str):
    return np.eye(RATINGS_CARDINALITY, dtype=int)[int(float(rating_str)) - 1]


TEST_FRAC = 0.2
fairness_reweighing_enabled = True


def transform_pd_X(df:pd.DataFrame, inp_cols:List[str]):
    """Original dataframe will be modified"""
    df[AGE] = df[AGE].apply(lambda age: [fix_age(age)])
    df[ZIP_CODE] = df[ZIP_CODE].apply(lambda zc: fix_zip_code(zc, n_digits=2, indexer=zip_code_indexer))
    df[FAVE_SPORTS] = df[FAVE_SPORTS].apply(fix_fav_sports_mlb)

    int_cols = [GENDER_F, GENDER_M, AD_NUM_FACES] +               AD_LABEL_COLS +               AD_SAFE_SEARCH_COLS +               SELECTED_HOMECOUNTRY_COLS +               SELECTED_INCOME_COLS +               SELECTED_MOSTLISTENEDMUSICS_COLS +               SELECTED_MOSTREADBOOKS_COLS +               SELECTED_MOSTWATCHEDMOVIES_COLS +               SELECTED_MOSTWATCHEDTVPROGRAMMES_COLS +               AD_OBJECT_COLS
    
    df[int_cols] = df[int_cols].applymap(lambda f: [int(f)])
    
    df["X"] = df[inp_cols].apply(np.concatenate, axis=1)
    # TODO: vectorize, else inefficient to sequentially loop over all examples
    X = np.array([x for x in df["X"]])
    return X


def merge_minority_classes(df, target_col):
    """Original dataframe is modified"""
    df.loc[df[target_col] != "1.0", target_col] = "2.0"
    return df


def transform_pd_y(df:pd.DataFrame, target_col:str):
    """Original dataframe will be modified"""
    if (RATINGS_CARDINALITY == 2) :
        df = merge_minority_classes(df, target_col)
        
    df["y"] = df[target_col].apply(create_target_pd)
    # TODO: vectorize, else inefficient to sequentially loop over all examples
    y = np.array([y for y in df["y"]])
    return y


def create_dataset_pd(inp_cols:List[str]=SELECTED_INP_COLS, target_col:str=TARGET_COL, fraction:float=1, test_frac:float=TEST_FRAC) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
    """
    Prepare the dataset for training on a fraction of all input data
    Columns using embeddings are split seperately and returned in list of tuples called embed_features
    """
    # NOTE: RANDOM_SEED should be same for both splits
    # Create (train, test) split of selected columns and target 
    df = ad_dataset_pd(SELECTED_COLS + EMBED_COLS).sample(frac=fraction)
    
    sample_weights = np.ones(shape=(df.shape[0],)) # equal weight to all examples
    if fairness_reweighing_enabled:
        bias_cols = [GENDER_F, AGE]
        sample_weights = reweigh_biased_cols(bias_cols,df)
    
    X, y = transform_pd_X(df[SELECTED_COLS], inp_cols), transform_pd_y(df[SELECTED_COLS], target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=RANDOM_SEED)
    
    # Create (train, test) split for each embedding column
    embed_features, embedding_store = {}, create_embedding_data(df)
    # TODO: this loop can be vectorized for speed
    for embed_col in EMBED_COLS:
        X_embed_train, X_embed_test = train_test_split(embedding_store[embed_col]["padded_col"], test_size=test_frac, random_state=RANDOM_SEED)
        embed_features[embed_col] = {"train": X_embed_train, "test": X_embed_test}
        
    sample_weights_train, _ = train_test_split(sample_weights, test_size=test_frac, random_state=RANDOM_SEED)
        
    return X_train, X_test, y_train, y_test, embed_features, embedding_store, sample_weights_train


from tensorboard import notebook


get_ipython().run_line_magic('reload_ext', 'tensorboard')


get_ipython().run_line_magic('tensorboard', '--logdir logs --port 6006')


notebook.list()


get_ipython().run_cell_magic('time', '', '\nX_train, X_test, y_train, y_test, embed_features, embedding_store, sample_weights_train = create_dataset_pd()')


inp = X_train, X_test, y_train, y_test, embed_features, embedding_store, sample_weights_train

with open(f"inp-{RANDOM_SEED}.pickle", "wb") as f:
    pickle.dump(inp, f)


train_embed_feature_1 = embed_features[FAV]["train"]
train_embed_feature_2 = embed_features[UNFAV]["train"]


def balance_classes(X_train , y_train , train_embed_feature_1 , train_embed_feature_2 , smote):
    
    if fairness_reweighing_enabled:
        raise ValueError('Dont use class balancing with reweighing enabled (fairness_reweighing_enabled = 1)')
        
    #concatenate embedding columns to X_train
    X_train_cols = X_train.shape[1]
    train_embed_feature_1_cols = train_embed_feature_1.shape[1]
    train_embed_feature_2_cols = train_embed_feature_2.shape[1]
    
    #convert OHE target to normal
    y_train_normal = [np.where(r==1)[0][0] for r in y_train]
    
    x_train_concat = np.concatenate((X_train,train_embed_feature_1,train_embed_feature_2),axis=1)
    
    if(smote):
        smote_oversample = SMOTE()
        X_t, y_t = smote_oversample.fit_resample(x_train_concat, y_train_normal)
    else:
        random_oversample = RandomOverSampler()
        X_t, y_t = random_oversample.fit_sample(x_train_concat, y_train_normal)
    
    #regenerate x_train , y_train and embedding columns from X_t
    y_train = np.eye(RATINGS_CARDINALITY, dtype=int)[y_t]
    X_train = X_t[:,0:X_train_cols]
    train_embed_feature_1 = X_t[:, X_train_cols : X_train_cols+train_embed_feature_1_cols]
    train_embed_feature_2 = X_t[:, X_train_cols+train_embed_feature_1_cols : X_train_cols+train_embed_feature_1_cols+train_embed_feature_2_cols]
    
    return X_train, y_train, train_embed_feature_1, train_embed_feature_2


if not fairness_reweighing_enabled:
    X_train, y_train, embed_features[FAV]["train"], embed_features[UNFAV]["train"] = balance_classes(X_train,y_train,
                                                                                  train_embed_feature_1,train_embed_feature_2,
                                                                                  True)


X_train.shape, y_train.shape, embed_features[FAV]["train"].shape, embed_features[UNFAV]["train"].shape


pd.DataFrame(y_train)[0].value_counts(), pd.DataFrame(y_test)[0].value_counts()


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


print("Size of train cat & num features ",X_train.shape)
print("Size of output for train ",y_train.shape)
print("Size of test cat & num features ",X_test.shape)
print("Size of output for test ",y_test.shape)
print("No. of embedded features ",len(embed_features))


embedding_store


def log_unknown_word_count(col:str):
    unk_embed_vec = np.zeros((embedding_store[col]["vocab_size"], WORD_VEC_DIMENSIONS))
    num_rows = np.count_nonzero(embedding_store[col]["embed_matrix"] == unk_embed_vec, axis=1).sum() / WORD_VEC_DIMENSIONS
    logging.warning(f"Could't find embeddings for {int(num_rows)} words in {col} column")
    
log_unknown_word_count(FAV), log_unknown_word_count(UNFAV) 


def create_embed_flat_layer(col:str, trainable_embed:bool=False):
    col_input = Input(shape=(embedding_store[col]['maxlen'],), name=col)
    col_embedded = Embedding(embedding_store[col]['vocab_size'], WORD_VEC_DIMENSIONS, weights=[embedding_store[col]['embed_matrix']],
                     input_length = embedding_store[col]['maxlen'], trainable=trainable_embed)(col_input)
    col_embedded_flat = Flatten()(col_embedded)
    return col_input, col_embedded_flat


def model_arch(inp):
    out = Dense(20, activation='relu')(inp)
    out = Dense(10, activation='relu')(out)
    return Dense(RATINGS_CARDINALITY, activation='softmax')(out) 


def create_model_with_embeddings():
    # Input layers
    selected_cols_input = Input(shape=(X_train.shape[1],), name="non_embed_inputs")
    fav_input, fav_embed = create_embed_flat_layer(FAV)
    unfav_input, unfav_embed = create_embed_flat_layer(UNFAV)

    # Concatenate the layers
    concatenated = concatenate([fav_embed, unfav_embed, selected_cols_input]) 
    out = model_arch(concatenated)

    # Create the model
    return Model(
        inputs = [fav_input, unfav_input, selected_cols_input],
        outputs = out,
    )


def create_model_without_embeddings():
    selected_cols_input = Input(shape=(X_train.shape[1],), name="non_embed_inputs")
    out = model_arch(selected_cols_input)

    # Create the model
    return Model(
        inputs = selected_cols_input,
        outputs = out,
    )


def model_fit_data(with_embed:bool=True):
    d = {}
    if with_embed:
        d["X_train"] = [embed_features[FAV]["train"], embed_features[UNFAV]["train"], X_train]
        d["y_train"] = y_train
        d["val_data"] = ([embed_features[FAV]["test"], embed_features[UNFAV]["test"], X_test], y_test)
    else:
        d["X_train"] = X_train
        d["y_train"] = y_train
        d["val_data"] = (X_test, y_test)
    return d


model = create_model_with_embeddings()
model.compile(
    optimizer=tf.optimizers.Adam(
        learning_rate=0.003,
        clipvalue=0.5
    ), 
    loss="categorical_crossentropy",
    metrics=keras_model_metrics
)

model.summary()


BATCH_SIZE = 4096
EPOCHS = 300


logdir = Path("logs")/f"{RANDOM_SEED}" #datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    logdir, 
    histogram_freq=max(1, ceil(EPOCHS / 20)), # to control the amount of logging
#     embeddings_freq=epochs,
)
print(f"Logging tensorboard data at {logdir}")


# %%time

# mfd = model_fit_data(with_embed=True)
# train_histories.append(model.fit(
#     mfd["X_train"], mfd["y_train"],
#     batch_size=BATCH_SIZE,
#     epochs=3,
#     validation_data=mfd["val_data"],
#     callbacks=[tfdocs.modeling.EpochDots(), tensorboard_callback], 
#     verbose=0
# ))


get_ipython().run_cell_magic('time', '', '\ntrain_histories.append(model.fit(\n    {\n        FAV: embed_features[FAV]["train"],\n        UNFAV: embed_features[UNFAV]["train"],\n        "non_embed_inputs": X_train\n    }, \n    y_train,\n    batch_size=BATCH_SIZE,\n    epochs=EPOCHS,\n    validation_data=(\n        {\n            FAV: embed_features[FAV]["test"],\n            UNFAV: embed_features[UNFAV]["test"],\n            "non_embed_inputs": X_test\n        },\n        y_test\n    ),\n    callbacks=[tfdocs.modeling.EpochDots(), tensorboard_callback], \n    verbose=0,\n    sample_weight=sample_weights_train #from af360 fairness reweighting\n))')


metrics_df = pd.DataFrame(train_histories[-1].history) # pick the latest training history

metrics_df.tail(1) # pick the last epoch's metrics


from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score, f1_score
import sklearn
from collections import OrderedDict

assert sklearn.__version__.startswith('0.22'), "Please upgrade scikit-learn (https://scikit-learn.org/stable/install.html)"


y_prob = model.predict([embed_features[FAV]["test"], embed_features[UNFAV]["test"], X_test], BATCH_SIZE)
y_true = y_test
y_pred = (y_prob / np.max(y_prob, axis=1).reshape(-1, 1)).astype(int) # convert probabilities to predictions


pd.DataFrame(OrderedDict({
    "macro_roc_auc_ovo": [roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro")],
    "weighted_roc_auc_ovo": roc_auc_score(y_test, y_prob, multi_class="ovo", average="weighted"),
    "macro_roc_auc_ovr": roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"),
    "weighted_roc_auc_ovr": roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"),
    "weighted_precision": precision_score(y_test, y_pred, average="weighted"),
    "weighted_recall": recall_score(y_test, y_pred, average="weighted"),
    "weighted_f1": f1_score(y_test, y_pred, average="weighted")
}))


print(classification_report(y_true, y_pred))


model.save((logdir/"keras_saved_model").as_posix(), save_format="tf")


PREDICTED_RATING, PREDICTION_CONFIDENCE = "pred_rating", "pred_confidence"


def predict_on_dataset(df:pd.DataFrame, model:tf.keras.Model, es:Dict, inp_cols_of_interest:List[str]):
    """
    Make predictions on df using the model and return inp_cols_of_interest
    IMPORTANT: embedding store, es should be the same as the model was trained on
    """
    X = transform_pd_X(df[SELECTED_COLS], SELECTED_INP_COLS)

    embed_features = {}
    for col in EMBED_COLS:
        embed_features[col], _ = transform_embed_col(df[col], es[col]["tokenizer"], es[col]["maxlen"])
     
    predict_proba = model.predict([embed_features[FAV], embed_features[UNFAV], X], batch_size=BATCH_SIZE)  
    df[PREDICTED_RATING], df[PREDICTION_CONFIDENCE] = np.argmax(predict_proba, axis=1), np.max(predict_proba, axis=1)
    
    return df[inp_cols_of_interest + [RATING, PREDICTED_RATING, PREDICTION_CONFIDENCE]]


PredictionReport = namedtuple("PredictionReport", "probabilities predicted_rating confidence")

# Note: Create a dataframe with all SELECTED_INP_COLS
test_df = pd.DataFrame({
    AGE: ["45"],
    ZIP_CODE: ["94086"],
    FAVE_SPORTS: ["I do not like Sports"]
})

probabilities = model.predict(transform_pd_X(test_df, SELECTED_INP_COLS))
predicted_rating, confidence = np.argmax(probabilities), np.max(probabilities)

PredictionReport(probabilities, predicted_rating, confidence)


RANDOM_SEED = 1589009535 # seed used in a historical run of interest for fairness analysis


with open(f"inp-{RANDOM_SEED}.pickle", "rb") as f:
    X_train, X_test, y_train, y_test, embed_features, embedding_store, sample_weights_train = pickle.load(f)
    
X_train.shape, y_train.shape, embed_features[FAV]["train"].shape, embed_features[UNFAV]["train"].shape


model = tf.keras.models.load_model(f"logs/{RANDOM_SEED}/keras_saved_model")
model.summary()


fairness_df = ad_dataset_pd() #.sample(n=100)
merge_minority_classes(fairness_df, RATING) # modifies original dataframe
fairness_df[RATING] = fairness_df[RATING].astype("float")
fairness_df[RATING] = fairness_df[RATING] - 1


train_fairness_df, val_fairness_df = train_test_split(fairness_df, test_size=TEST_FRAC, random_state=RANDOM_SEED)
train_fairness_df.shape, val_fairness_df.shape


get_ipython().run_cell_magic('time', '', '\nres = predict_on_dataset(val_fairness_df, model, embedding_store, [AGE, GENDER, INCOME, HOME_COUNTRY])')


out_path = f"logs/{RANDOM_SEED}/inference_data.csv"

res.to_csv(out_path, index=False)
print(f"Saved inference data at {out_path}")


# Credits: https://stackoverflow.com/a/50671617/1585523 has good explanations for all metrics and is from where the code has been copied

def metrics_from_df(df:pd.DataFrame, confidence_threshold=0):
    """Drop examples with probability < confidence_threshold from calc"""
    y_true = df[RATING]
    y_pred = df[PREDICTED_RATING]
    cnf_matrix = confusion_matrix(y_true, y_pred)
    
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    
    return {
        "TPR": TPR, "TNR": TNR, "PPV": PPV, "NPV": NPV, "FPR": FPR, "FNR": FNR, "FDR": FDR, "ACC": ACC
    }


class GroupFairnessMetrics:
    def __init__(self, model_inference_data:pd.DataFrame, protected_feature:str):
        """
        Compute fairness metrics between 2 groups of population - privileged & unprivileged
        Based on your dataset you could use,
            "Female" to privileged_grp_label and 
            "Male" to be unprivileged_grp_label
        for Gender as the protected_feature
        All metrics are computed on model_inference_data which has RATING and PREDICTED_RATING values for each row
        """
        self._df, self._pf, = model_inference_data, protected_feature
        self._base_metrics = "fairness_metrics_per_class"
        self._pf_metrics_df = self._df.groupby(self._pf).apply(metrics_from_df).to_frame(self._base_metrics)
    
    def fetch_base_metrics(self):
        return self._pf_metrics_df
    
    def equal_opportunity_difference(self, pg_lbl:str, upg_lbl:str, rating_class=1):
        r"""TPR{unprivileged} - TPR{privileged} ideally should be zero"""
        upg_opp = self._pf_metrics_df.loc[upg_lbl][self._base_metrics]["TPR"][rating_class]
        pg_opp = self._pf_metrics_df.loc[pg_lbl][self._base_metrics]["TPR"][rating_class]
        return upg_opp - pg_opp

    def statistical_parity_difference(self):
        raise NotImplementedError("TODO")

    def average_odds_difference(self, pg_lbl:str, upg_lbl:str, rating_class=1):
        """Average of difference in FPR and TPR for unprivileged and privileged groups"""
        tpr_diff = self.equal_opportunity_difference(pg_lbl, upg_lbl, rating_class)
        
        upg_fpr = self._pf_metrics_df.loc[upg_lbl][self._base_metrics]["FPR"][rating_class]
        pg_fpr = self._pf_metrics_df.loc[pg_lbl][self._base_metrics]["FPR"][rating_class]
        fpr_diff = upg_fpr - pg_fpr
        
        return 0.5 * (fpr_diff + tpr_diff)

    def disparate_impact():
        raise NotImplementedError("TODO")

    def theil_index():
        raise NotImplementedError("TODO")


def plot_for_metric_class(metric_df:pd.DataFrame, metric:str="FPR", rating_class:int=1):
    """Generates plot for metric and given rating_class from metric_df indexed by dimension of interest"""
    plot_df = metric_df.apply(lambda m: m["fairness_metrics_per_class"][metric][rating_class], axis=1)
    plot_df = plot_df.reset_index().rename({0: metric}, axis=1)
    return plot_df


res = pd.read_csv(f"logs/{RANDOM_SEED}/inference_data.csv")


res.sample(4).T


gender_fairness_metrics = GroupFairnessMetrics(res, GENDER)


plot_df = plot_for_metric_class(gender_fairness_metrics.fetch_base_metrics())
plot_df


ax = sns.barplot(x=GENDER, y="FPR", data=plot_df)


plot_df.to_clipboard(False)


gender_fairness_metrics.equal_opportunity_difference("F", "M")


gender_fairness_metrics.average_odds_difference("F", "M")


res[AGE] = res[AGE].astype("int")


ax = sns.distplot(res[AGE], kde=False, bins=50)


AGE_BUCKET = AGE + "_bucket"
bucket_boundaries = [0, 20, 40, 100] # refer pandas.cut() for syntax on binning
age_labels = ["young", "middle-age", "old"] # refer pandas.cut() for syntax on labels


res[AGE_BUCKET] = pd.cut(res[AGE], bins=bucket_boundaries, labels=age_labels)
res[[AGE, AGE_BUCKET]].sample(n=5)


ax = sns.countplot(res[AGE_BUCKET])
res[AGE_BUCKET].value_counts()


age_metrics_df = res.groupby(AGE_BUCKET).apply(metrics_from_df).to_frame("fairness_metrics_per_class")
age_metrics_df


plot_df = plot_for_metric_class(age_metrics_df)
plot_df


ax = sns.barplot(x=AGE_BUCKET, y="FPR", data=plot_df)


plot_df.to_clipboard(False)


age_gfm = GroupFairnessMetrics(res, AGE_BUCKET)


age_gfm.equal_opportunity_difference("young", "middle-age"), age_gfm.equal_opportunity_difference("young", "old")


age_gfm.average_odds_difference("young", "middle-age"), age_gfm.average_odds_difference("young", "old")


ax = sns.countplot(res[INCOME])
res[INCOME].value_counts()


income_metrics_df = res.groupby(INCOME).apply(metrics_from_df).to_frame("fairness_metrics_per_class")
income_metrics_df


plot_df = plot_for_metric_class(income_metrics_df)
plot_df


ax = sns.barplot(x=INCOME, y="FPR", data=plot_df)


plot_df.to_clipboard(False)


ax = sns.countplot(res[HOME_COUNTRY])
res[HOME_COUNTRY].value_counts()


homecountry_metrics_df = res.groupby(HOME_COUNTRY).apply(metrics_from_df).to_frame("fairness_metrics_per_class")
homecountry_metrics_df


plot_df = plot_for_metric_class(homecountry_metrics_df)
plot_df


ax = sns.barplot(x=HOME_COUNTRY, y="FPR", data=plot_df)


# Filter train & val df for Ads of Category 11 - "Kitchen & Home"
train_fairness_df_adcat11 = train_fairness_df[train_fairness_df['AdId'].str.startswith("A11_")]
val_fairness_df_adcat11 = val_fairness_df[val_fairness_df['AdId'].str.startswith("A11_")]
train_fairness_df_adcat11.shape, val_fairness_df_adcat11.shape


val_fairness_df_adcat11[GENDER].value_counts()


val_fairness_df[GENDER].value_counts()


get_ipython().run_cell_magic('time', '', '\nres = predict_on_dataset(val_fairness_df_adcat11, model, embedding_store, [AGE, GENDER, INCOME, HOME_COUNTRY])')


out_path = f"logs/{RANDOM_SEED}/inference_data_AdCat11.csv"

res.to_csv(out_path, index=False)
print(f"Saved inference data at {out_path}")


out_path


# res = pd.read_csv("logs/" + str(RANDOM_SEED) + "/inference_data.csv")
res = pd.read_csv(out_path)


res.sample(4).T


gender_metrics_df = res.groupby(GENDER).apply(metrics_from_df).to_frame("fairness_metrics_per_class")
gender_metrics_df


plot_df = plot_for_metric_class(gender_metrics_df)
plot_df


ax = sns.barplot(x=GENDER, y="FPR", data=plot_df)


plot_df.to_clipboard(False)


ax = sns.countplot(res[INCOME])
res[INCOME].value_counts()


income_metrics_df = res.groupby(INCOME).apply(metrics_from_df).to_frame("fairness_metrics_per_class")
income_metrics_df


plot_df = plot_for_metric_class(income_metrics_df)
plot_df


ax = sns.barplot(x=INCOME, y="FPR", data=plot_df)


plot_df.to_clipboard(False)


# Filter train & val df for Ads of Category 11 - "Kitchen & Home"
train_fairness_df_adcat03 = train_fairness_df[train_fairness_df['AdId'].str.startswith("A03_")]
val_fairness_df_adcat03 = val_fairness_df[val_fairness_df['AdId'].str.startswith("A03_")]
train_fairness_df_adcat03.shape, val_fairness_df_adcat03.shape


val_fairness_df_adcat03[GENDER].value_counts()


val_fairness_df[GENDER].value_counts()


get_ipython().run_cell_magic('time', '', '\nres = predict_on_dataset(val_fairness_df_adcat03, model, embedding_store, [AGE, GENDER, INCOME, HOME_COUNTRY])')


out_path = f"logs/{RANDOM_SEED}/inference_data_AdCat03.csv"

res.to_csv(out_path, index=False)
print(f"Saved inference data at {out_path}")


out_path


# res = pd.read_csv("logs/" + str(RANDOM_SEED) + "/inference_data.csv")
res = pd.read_csv(out_path)


res.sample(4).T


gender_metrics_df = res.groupby(GENDER).apply(metrics_from_df).to_frame("fairness_metrics_per_class")
gender_metrics_df


plot_df = plot_for_metric_class(gender_metrics_df)
plot_df


ax = sns.barplot(x=GENDER, y="FPR", data=plot_df)


plot_df.to_clipboard(False)


# Filter train & val df for Ads of Category 11 - "Kitchen & Home"
train_fairness_df_adcat06 = train_fairness_df[train_fairness_df['AdId'].str.startswith("A06_")]
val_fairness_df_adcat06 = val_fairness_df[val_fairness_df['AdId'].str.startswith("A06_")]
train_fairness_df_adcat06.shape, val_fairness_df_adcat06.shape


val_fairness_df_adcat06[GENDER].value_counts()


val_fairness_df[GENDER].value_counts()


get_ipython().run_cell_magic('time', '', '\nres = predict_on_dataset(val_fairness_df_adcat06, model, embedding_store, [AGE, GENDER, INCOME, HOME_COUNTRY])')


out_path = f"logs/{RANDOM_SEED}/inference_data_AdCat06.csv"

res.to_csv(out_path, index=False)
print(f"Saved inference data at {out_path}")


out_path


# res = pd.read_csv("logs/" + str(RANDOM_SEED) + "/inference_data.csv")
res = pd.read_csv(out_path)


res.sample(4).T


gender_metrics_df = res.groupby(GENDER).apply(metrics_from_df).to_frame("fairness_metrics_per_class")
gender_metrics_df


plot_df = plot_for_metric_class(gender_metrics_df)
plot_df


ax = sns.barplot(x=GENDER, y="FPR", data=plot_df)


plot_df.to_clipboard(False)


res[AGE] = res[AGE].astype("int")


ax = sns.distplot(res[AGE], kde=False, bins=50)


AGE_BUCKET = AGE + "_bucket"
bucket_boundaries = [0, 20, 40, 100] # refer pandas.cut() for syntax on binning
age_labels = ["young", "middle-age", "old"] # refer pandas.cut() for syntax on labels


res[AGE_BUCKET] = pd.cut(res[AGE], bins=bucket_boundaries, labels=age_labels)
res[[AGE, AGE_BUCKET]].sample(n=5)


ax = sns.countplot(res[AGE_BUCKET])
res[AGE_BUCKET].value_counts()


age_metrics_df = res.groupby(AGE_BUCKET).apply(metrics_from_df).to_frame("fairness_metrics_per_class")
age_metrics_df


plot_df = plot_for_metric_class(age_metrics_df)
plot_df


ax = sns.barplot(x=AGE_BUCKET, y="FPR", data=plot_df)


plot_df.to_clipboard(False)


ax = sns.countplot(res[INCOME])
res[INCOME].value_counts()


income_metrics_df = res.groupby(INCOME).apply(metrics_from_df).to_frame("fairness_metrics_per_class")
income_metrics_df


plot_df = plot_for_metric_class(income_metrics_df)
plot_df


ax = sns.barplot(x=INCOME, y="FPR", data=plot_df)


plot_df.to_clipboard(False)


EXAMPLE_BATCH = next(iter(input_fn_train(3)))[0]


EXAMPLE_BATCH


def test_feature_column(feature_column):
    feature_layer = tf.keras.layers.DenseFeatures(feature_column)
    return feature_layer(EXAMPLE_BATCH).numpy()


age_fc = tf.feature_column.numeric_column(AGE, normalizer_fn=lambda x: (x - MEAN_AGE) / STD_AGE)


zip_fcs = [
    tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            f"{ZIP_CODE}{i}", vocabulary_list=list(string.digits), 
            num_oov_buckets=1)
    )
    for i in range(FIRST_K_ZIP_DIGITS)
]


EXAMPLE_BATCH[AGE], test_feature_column(age_fc)


{k: v for k, v in EXAMPLE_BATCH.items() if k.startswith(ZIP_CODE)}, test_feature_column(zip_fcs)


tf.keras.layers.concatenate(age_fc, zip_fcs[0])


import os
import tempfile
import apache_beam as beam
import numpy as np
import pandas as pd
from datetime import datetime

import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_data_validation as tfdv
from tensorflow_model_analysis.addons.fairness.post_export_metrics import fairness_indicators
from tensorflow_model_analysis.addons.fairness.view import widget_view
from fairness_indicators.examples import util

from witwidget.notebook.visualization import WitConfigBuilder
from witwidget.notebook.visualization import WitWidget


logdir


y_train[:100, :].shape


from collections import OrderedDict


def dl():
    for i in range(100):
        yield OrderedDict({
            FAV: embed_features[FAV]["train"][i, :],
            UNFAV: embed_features[UNFAV]["train"][i, :],
            "non_embed_inputs": X_train[i, :],
            RATING: y_train[i, :]
        })


dataset = tf.data.Dataset.from_generator(dl, 
            (tf.float32, tf.float32, tf.float32, tf.float32),
            (tf.TensorShape([40]), tf.TensorShape([34]), tf.TensorShape([308]), tf.TensorShape([2])))


NON_EMBED_INPUTS = "non_embed_inputs"


dataset = tf.data.Dataset.from_generator(dl, 
            {FAV: tf.float32, UNFAV: tf.float32, NON_EMBED_INPUTS: tf.float32, RATING: tf.float32},
            {FAV: tf.TensorShape([40]), UNFAV: tf.TensorShape([34]), NON_EMBED_INPUTS: tf.TensorShape([308]), RATING: tf.TensorShape([2])})


fields_to_save = [FAV, UNFAV, NON_EMBED_INPUTS]


def serialize_example(*example:List[tf.Tensor]):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible data type.
    fields = {field: dtype_feature_map[example[i].dtype](example[i]) for i, field in enumerate(fields_to_save)}

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=fields))
    return example_proto.SerializeToString()

def tf_serialize_example(example:Dict):
    tf_string = tf.py_function(
        serialize_example,
        [example[field] for field in fields_to_save],  # pass these args to the above function.
        tf.string)      # the return type is `tf.string`.
    return tf.reshape(tf_string, ()) # The result is a scalar


# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    if len(value.shape) > 0:
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if len(value.shape) > 0:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


dtype_feature_map = {
    tf.dtypes.string: _bytes_feature,
    tf.dtypes.float16: _float_feature,
    tf.dtypes.float32: _float_feature,
    tf.dtypes.float64: _float_feature,
    tf.dtypes.int8: _int64_feature,
    tf.dtypes.int16: _int64_feature,
    tf.dtypes.int32: _int64_feature,
    tf.dtypes.int64: _int64_feature
}


analysis_data_folder = Path("./analysis_data_with_temporal_as_ints")
analysis_train_data_folder = analysis_data_folder/"train.tfrecords"
analysis_val_data_folder = analysis_data_folder/"val.tfrecords"

analysis_data_folder.mkdir(exist_ok=True, parents=True)


from shutil import rmtree


def save_dataset_part(dataset:tf.data, part:int, save_in_folder:Union[Path, str], overwrite=False):
    """
    Save a sharded dataset to a folder
    """
    print(f"Working on part-{part} of the file")
    save_in_folder:Path = Path(save_in_folder) # change to Path object if not already
    overwrite and rmtree(save_in_folder, ignore_errors=True)
    save_in_folder.mkdir(parents=True, exist_ok=True)
    writer = tf.data.experimental.TFRecordWriter(f"{save_in_folder}/part-{part}")
    writer.write(dataset)
    
save_dataset_part_parallel_fn = partial(save_dataset_part, save_in_folder=analysis_data_folder)


def prepare_dataset_for_save(dataset):
    return dataset.        unbatch().        map(tf_serialize_example, tf.data.experimental.AUTOTUNE)


d1 = prepare_dataset_for_save(dataset.batch(1))


save_dataset_part(d1, 0, analysis_train_data_folder, overwrite=True)


get_ipython().system(' ls {logdir}')


#@title Fairness Indicators Computation Options
tfma_eval_result_path = f'{logdir}/tfma_eval_result'

#@markdown Modify the slice_selection for experiments on other identities.
slice_selection = 'gender' #@param ["sexual_orientation", "gender", "religion", "race", "disability"]
#@markdown Confidence Intervals can help you make better decisions regarding your data, but as it requires computing multiple resamples, is slower particularly in the colab environment that cannot take advantage of parallelization.
compute_confidence_intervals = False #@param {type:"boolean"}

# # Define slices that you want the evaluation to run on.
slice_spec = [
    tfma.slicer.SingleSliceSpec(), # Overall slice
    tfma.slicer.SingleSliceSpec(columns=[slice_selection]),
]

# # Add the fairness metrics.
add_metrics_callbacks = [
  tfma.post_export_metrics.fairness_indicators(
      thresholds=[0.1, 0.3, 0.5, 0.7, 0.9],
      labels_key=RATING
      )
]

eval_shared_model = tfma.default_eval_shared_model(
    eval_saved_model_path=(logdir/"keras_saved_model").as_posix(),
    add_metrics_callbacks=add_metrics_callbacks)


from google.protobuf import text_format


eval_config = text_format.Parse("""
  # https://www.tensorflow.org/tfx/model_analysis/setup#model_specs
  model_specs {
    # This assumes a serving model with a "serving_default" signature.
    label_key: "%(label_key)s"
  }
  metrics_specs {
    metrics { class_name: "ExampleCount" }
    metrics { class_name: "WeightedExampleCount" }
    metrics { class_name: "SparseCategoricalCrossentropy" }
    metrics { class_name: "SparseCategoricalAccuracy" }
    metrics { class_name: "Precision" config: '"top_k": 1' }
    metrics { class_name: "Precision" config: '"top_k": 3' }
    metrics { class_name: "Recall" config: '"top_k": 1' }
    metrics { class_name: "Recall" config: '"top_k": 3' }
    metrics { class_name: "AUC" }
  }
  # https://www.tensorflow.org/tfx/model_analysis/setup#slicing_specs
  slicing_specs {
    feature_keys: ["Age"]
  }
""" % {"label_key": RATING}, tfma.EvalConfig())

eval_shared_model = tfma.default_eval_shared_model(
    eval_saved_model_path=(logdir/"keras_saved_model").as_posix(), 
    eval_config=eval_config, add_metrics_callbacks=add_metrics_callbacks)


eval_result = tfma.run_model_analysis(eval_shared_model=eval_shared_model,
                                          eval_config=eval_config,
                                          data_location=(analysis_train_data_folder/"*").as_posix(),
                                          file_format='tfrecords',
#                                           slice_spec=slice_list,
#                                           output_path='sample_data',
#                                           extractors=None, 
                                          desired_batch_size=1024,
                                          random_seed_for_testing=RANDOM_SEED)


slices = [tfma.slicer.SingleSliceSpec(columns=[AGE])]
tfma.view.render_slicing_metrics(eval_result, slicing_spec=slices[0])


validate_tf_file = "analysis_data_with_temporal_as_ints/train.tfrecords/part-0"


# Run the fairness evaluation.
with beam.Pipeline() as pipeline:
    _ = (
      pipeline
      | 'ReadData' >> beam.io.ReadFromTFRecord(validate_tf_file)
      | 'ExtractEvaluateAndWriteResults' >>
       tfma.ExtractEvaluateAndWriteResults(
                 eval_shared_model=eval_shared_model,
                 slice_spec=slice_spec,
                 compute_confidence_intervals=compute_confidence_intervals,
                 output_path=tfma_eval_result_path)
    )

#     eval_result = tfma.load_eval_result(output_path=tfma_eval_result_path)


logdir


def create_target_for_model(d:Dict, target_col:str=RATING) -> Tuple[Dict, np.array]:
    target = d.pop(target_col)
    return d, target


for d in dataset.batch(1).take(2):
    pprint(d)


for d in dataset.map(create_dataset_pd).take(2):
    pprint(d)


for d in dl():
    for k in d:
        print(k, d[k].shape)
    break


tf.data.Dataset.from_generator()


num_samples = 100

pd.DataFrame(
    {
        FAV: embed_features[FAV]["train"][:num_samples, :],
        UNFAV: embed_features[UNFAV]["train"][:num_samples, :],
        "non_embed_inputs": X_train[:num_samples, :],
        RATING: y_train[:num_samples, :]
    },
    index=range(num_samples)
)


beam.io.ReadF


get_ipython().run_line_magic('pinfo2', 'eval_shared_model')


get_ipython().run_line_magic('pinfo2', 'tfma.ExtractEvaluateAndWriteResults')


get_ipython().system(' ./setup.sh # uncomment if you wish to install any new packages')


from aif360.datasets import GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing


from aif360.metrics import DatasetMetric


import aif360.sklearn.metrics as fm


fm










