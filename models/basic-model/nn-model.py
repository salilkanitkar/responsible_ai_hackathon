#!/usr/bin/env python
# coding: utf-8

get_ipython().system(' ./setup.sh # uncomment if you wish to install any new packages')


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

import chakin
import json
import os
from collections import defaultdict
import zipfile
import sqlite3
import logging
from tempfile import TemporaryDirectory
from fastprogress import progress_bar

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import concatenate

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


def ad_dataset_pd(usecols:List[str]):
    """
    Read from csv files given set of columns into Pandas Dataframe
    """
    return pd.read_csv(users_ads_rating_csv,usecols=usecols, dtype=str)


ad_dataset_pd(SELECTED_COLS).sample(5).T


chakin.search(lang='English')


WORD_VEC_DIMENSIONS = 100


get_ipython().run_cell_magic('time', '', '\nembedding_index = EmbeddingFactory(Path("./embeddings/"), "GloVe.6B.100d", WORD_VEC_DIMENSIONS, "embeddings/glove-wiki", nrows=None, skiprows=None)')


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


CACHE = defaultdict(dict) # Store matrix and metadata for each embedding column for later use


def transform_embed_cols(df:pd.DataFrame, embed_col:str):
    """
    Takes dataframe and column name and transforms column and stores
    vocab size, max len & embedding matrix into CACHE
    """
    # Input dataframe and embed_col
    t = Tokenizer()
    t.fit_on_texts(df[embed_col])

    # UNK added as tokenizer starts indexing from 1
    words = ["UNK"] + list(t.word_index.keys()) # in order of tokenizer
    CACHE[embed_col]["vocab_size"] = len(words)

    # integer encode the text data
    encoded_col = t.texts_to_sequences(df[embed_col])

    # calculate max len of vector and make length equal by padding with zeros
    maxlen = max(len(x) for x in encoded_col) 
    CACHE[embed_col]["maxlen"] = maxlen

    padded_col = pad_sequences(encoded_col, maxlen=maxlen, padding='post')

    # create a weight matrix
    embeddings = dict.fromkeys(words, " ".join(["0"] * WORD_VEC_DIMENSIONS)) # default embeddings to all words as 0
    embeddings.update(dict(embedding_index.fetch_word_vectors(words))) # update for known words
    # reorder to match tokenizer's indexing
    emb_matrix = pd.DataFrame.from_dict(embeddings, orient="index").loc[words, 0].str.split(" ", expand=True).to_numpy().astype(np.float16)
    CACHE[embed_col]["embed_matrix"] = emb_matrix
    assert emb_matrix.shape[0] == len(words), "Not all words have embeddings"
    
    return padded_col, t


p, _ = transform_embed_cols(ad_dataset_pd([FAV]).sample(n=5), FAV)
p


CACHE, CACHE[FAV]["embed_matrix"].shape


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


RATINGS_CARDINALITY = 5 # not zero based indexing i.e. ratings range from 1 to 5


def create_target_pd(rating_str:str):
    return np.eye(RATINGS_CARDINALITY, dtype=int)[int(float(rating_str)) - 1]


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


def transform_pd_y(df:pd.DataFrame, target_col:str):
    """Original dataframe will be modified"""
    df["y"] = df[target_col].apply(create_target_pd)
    # TODO: vectorize, else inefficient to sequentially loop over all examples
    y = np.array([y for y in df["y"]])
    return y


def create_dataset_pd(inp_cols:List[str]=SELECTED_INP_COLS, target_col:str=TARGET_COL, fraction:float=1, test_frac:float=0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
    """
    Prepare the dataset for training on a fraction of all input data
    Columns using embeddings are split seperately and returned in list of tuples called embed_features
    """
    # NOTE: RANDOM_SEED should be same for both splits
    # Create (train, test) split of selected columns and target 
    df = ad_dataset_pd(SELECTED_COLS + EMBED_COLS).sample(frac=fraction)
    
    X, y = transform_pd_X(df[SELECTED_COLS], inp_cols), transform_pd_y(df[SELECTED_COLS], target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=RANDOM_SEED)
    
    # Create (train, test) split for each embedding column
    embed_features = {}
    for embed_col in EMBED_COLS:
        X_embed_col, tokenizer = transform_embed_cols(df, embed_col)
        X_embed_train, X_embed_test = train_test_split(X_embed_col, test_size=test_frac, random_state=RANDOM_SEED)
        embed_features[embed_col] = {"train": X_embed_train, "test": X_embed_test}
        
    return X_train, X_test, y_train, y_test, embed_features


from tensorboard import notebook


get_ipython().run_line_magic('reload_ext', 'tensorboard')


get_ipython().run_line_magic('tensorboard', '--logdir logs --port 6006')


notebook.list()


get_ipython().run_cell_magic('time', '', '\nX_train, X_test, y_train, y_test, embed_features = create_dataset_pd()')


print("Size of train cat & num features ",X_train.shape)
print("Size of output for train ",y_train.shape)
print("Size of test cat & num features ",X_test.shape)
print("Size of output for test ",y_test.shape)
print("No. of embedded features ",len(embed_features))


# Let's check what CACHE contains and check it's shape
CACHE


def log_unknown_word_count(col:str):
    unk_embed_vec = np.zeros((CACHE[col]["vocab_size"], WORD_VEC_DIMENSIONS))
    num_rows = np.count_nonzero(CACHE[col]["embed_matrix"] == unk_embed_vec, axis=1).sum() / WORD_VEC_DIMENSIONS
    logging.warning(f"Could't find embeddings for {int(num_rows)} words in {col} column")
    
log_unknown_word_count(FAV), log_unknown_word_count(UNFAV) 


def create_embed_flat_layer(col:str, trainable_embed:bool=False):
    col_input = Input(shape=(CACHE[col]['maxlen'],))
    col_embedded = Embedding(CACHE[col]['vocab_size'], WORD_VEC_DIMENSIONS, weights=[CACHE[col]['embed_matrix']],
                     input_length = CACHE[col]['maxlen'], trainable=trainable_embed)(col_input)
    col_embedded_flat = Flatten()(col_embedded)
    return col_input, col_embedded_flat


# Input layers
selected_cols_input = Input(shape=(X_train.shape[1],))
fav_input, fav_embed = create_embed_flat_layer(FAV)
unfav_input, unfav_embed = create_embed_flat_layer(UNFAV)


# Concatenate the layers

concatenated = concatenate([fav_embed, unfav_embed, selected_cols_input]) 
out = Dense(10, activation='relu')(concatenated)
out = Dense(RATINGS_CARDINALITY, activation='softmax')(out)


# Create the model
model = Model(
    inputs = [fav_input, unfav_input, selected_cols_input],
    outputs = out,
)


model.summary()


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


model.compile(
    optimizer=tf.optimizers.Adam(
        learning_rate=0.003,
        clipvalue=0.5
    ), 
    loss="categorical_crossentropy",
    metrics=keras_model_metrics
)


BATCH_SIZE = 4096
EPOCHS = 1000


logdir = Path("logs")/datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    logdir, 
    histogram_freq=max(1, ceil(EPOCHS / 20)), # to control the amount of logging
#     embeddings_freq=epochs,
)
print(f"Logging tensorboard data at {logdir}")


get_ipython().run_cell_magic('time', '', '\ntrain_histories.append(model.fit(\n    [embed_features[FAV]["train"], embed_features[UNFAV]["train"], X_train],\n    y_train,\n    batch_size=BATCH_SIZE,\n    epochs=EPOCHS,\n    validation_data=([embed_features[FAV]["test"], embed_features[UNFAV]["test"], X_test], y_test),\n    callbacks=[tfdocs.modeling.EpochDots(), tensorboard_callback], \n    verbose=0\n))')


metrics_df = pd.DataFrame(train_histories[-1].history) # pick the latest training history

metrics_df.tail(1) # pick the last epoch's metrics


from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score, f1_score
import sklearn
from collections import OrderedDict

assert sklearn.__version__ == '0.22.2.post1', "Please upgrade scikit-learn (https://scikit-learn.org/stable/install.html)"


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

