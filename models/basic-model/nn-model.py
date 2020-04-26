#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
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

print(f"Using Tensorflow, {tf.__version__} on Python interpreter, {sys.version_info}")


RANDOM_SEED = int(time.time())

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"Using random seed, {RANDOM_SEED}")


DATA_FOLDER = Path("../../dataset/")
BATCH_SIZE = 4096 # bigger the batch, faster the training but bigger the RAM needed
TARGET_COL = "Rating"

# data files path are relative DATA_FOLDER
users_ads_rating_csv = DATA_FOLDER/"AllUsers_Ads_Ratings_df.csv"


USER_ID = "UserId"
AD_ID = "AdId"
AGE = "Age"
ZIP_CODE = "Cap/Zip-Code"
COUNTRIES_VISITED = "Countries visited"
FAVE_SPORTS = "Fave Sports"
GENDER = "Gender"
HOME_COUNTRY = "Home country"
HOME_TOWN = "Home town"
INCOME = "Income"
LAST_NAME = "Last Name"
MOST_LISTENED_MUSICS = "Most listened musics"
MOST_READ_BOOKS = "Most read books"
MOST_VISITED_WEBSITES = "Most visited websites"
MOST_WATCHED_MOVIES = "Most watched movies"
MOST_WATCHED_TV_PROGRAMMES = "Most watched tv programmes"
NAME = "Name"
PAYPAL = "Paypal"
TIMEPASS = "Timepass"
TYPE_OF_JOB = "Type of Job"
WEEKLY_WORKING_HOURS = "Weekly working hours"
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
RATING = "Rating"

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
    RATING: "**"
}

SELECTED_COLS = [AGE, ZIP_CODE, FAVE_SPORTS, GENDER, HOME_COUNTRY, HOME_TOWN, INCOME, MOST_LISTENED_MUSICS, MOST_READ_BOOKS, 
                 MOST_VISITED_WEBSITES, MOST_WATCHED_MOVIES, MOST_WATCHED_TV_PROGRAMMES, TIMEPASS, TYPE_OF_JOB, WEEKLY_WORKING_HOURS, 
                 FAVE1, FAVE2, FAVE3, FAVE4, FAVE5, FAVE6, FAVE7, FAVE8, FAVE9, FAVE10, UNFAVE1, UNFAVE2, UNFAVE3, UNFAVE4, UNFAVE5, 
                 UNFAVE6, RATING]


def ad_dataset(batch_size=BATCH_SIZE, shuffle=True):
    return tf.data.experimental.make_csv_dataset(
        users_ads_rating_csv.as_posix(),
        batch_size,
        column_defaults={col:default for col, default in COL_DEFAULTS.items() if col in SELECTED_COLS},
        select_columns=list(SELECTED_COLS),
        label_name=None,
        shuffle=shuffle,
        shuffle_buffer_size=1000,
        shuffle_seed=RANDOM_SEED,
        sloppy=True,
        ignore_errors=False # set true while training if required
    )


for d in ad_dataset(3).take(1):
    pprint(d)


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
    return age

def fix_age_tf(example:Dict):
    """Wrap in a py_function for TF to run inside its execution graph"""
    example[AGE] = tf.py_function(fix_age, [example[AGE]], tf.int16)
    return example


fix_age("50"), fix_age("50.5"), fix_age("-10"), fix_age("bad_age_10"), fix_age("300")


for d in ad_dataset(1, True).map(fix_age_tf).batch(10).take(5):
    pprint(d[AGE])


DEFAULT_ZIP_CODE = "00000"


def fix_zip_code(zip_code:tf.string, n_digits) -> List[str]:
    """Extracts the the first n_digits as a list"""
    try:
        zip_code = zip_code.numpy()[0].decode('ascii') # very ineffecient way
        return list(zip_code.strip()[:n_digits])
    except:
        return list(DEFAULT_ZIP_CODE[:n_digits])

def fix_zip_code_tf(example:Dict, n_digits=2):
    """Creates new columns for the first n_digits in zip_code"""
    fix_zip_code_fn = partial(fix_zip_code, n_digits=n_digits)
    zip_digits = tf.py_function(fix_zip_code_fn, [example[ZIP_CODE]], [tf.string] * n_digits)
    zip_features = {}
    for i in range(n_digits):
        val = zip_digits[i]
        val.set_shape([]) # https://github.com/tensorflow/tensorflow/issues/24520#issuecomment-579421744
        zip_features[f"{ZIP_CODE}{i}"] = val
    example.update(zip_features)
    example.pop(ZIP_CODE)
    return example


(fix_zip_code(tf.constant([b"43556"], shape=(1,), dtype=tf.string), 10),
fix_zip_code(tf.constant([b"43556"], shape=(1,), dtype=tf.string), 2),
fix_zip_code(tf.constant([b"43556"], shape=(1,), dtype=tf.string), 4),
fix_zip_code(tf.constant([43556], shape=(1,), dtype=tf.int32), 4),\
fix_zip_code(None, 3))


for d in ad_dataset(1, True).map(fix_zip_code_tf).batch(5).take(3):
    pprint({k: v for k, v in d.items() if k.startswith(ZIP_CODE)})


tf.strings.split("a, b, c", r",")


def fix_fav_sports(sports_str:str, topk=2, pad_constant="PAD_SPORT") -> List[str]:
    sports = re.sub(r"\s*\(.*,.*\)\s*", "", sports_str.numpy()[0].decode('ascii')).split(r"\s*,\s*")[:topk]
    right_pad_width = max(0, topk - len(sports))
    tf.print(right_pad_width, len(sports))
    result = np.pad(sports, (0, right_pad_width), constant_values=pad_constant)
    tf.print(result)
    return result


def fix_fav_sports(sports_str:str, topk=2, pad_constant="PAD_SPORT") -> List[str]:
    sports = tf.strings.regex_replace(sports_str, r"\s*\(.*,.*\)\s*", "")
    sports = tf.strings.regex_replace(sports, r"\s*,\s*", ",")
    sports = tf.strings.split(sports, ",").numpy()[:topk]
    tf.print(sports.shape[0])
    right_pad_width = max(0, topk - sports.shape[0])
    result = np.pad(sports, (0, right_pad_width), constant_values=pad_constant) 
    return result


def fix_fav_sports_tf(example:Dict, topk=2, pad_constant="PAD_SPORT"):
    fix_fav_sports_fn = partial(fix_fav_sports, topk=topk, pad_constant=pad_constant)
#     example[FAVE_SPORTS] = tf.py_function(fix_fav_sports_fn, example[FAVE_SPORTS], [tf.string] * topk)
    example[FAVE_SPORTS] = fix_fav_sports_fn(example[FAVE_SPORTS])
    return example


fix_fav_sports("Individual sports (Tennis, Archery, ...), Indoor sports, Endurance sports, Skating sports")


for d in ad_dataset(1, True).map(fix_fav_sports_tf).batch(1).take(1):
    pprint(d[FAVE_SPORTS])


x["efs"] = x[["Fave Sports"]].apply(lambda r: re.findall(r"([\w\s]+\(.*,.*\))|([\w\s]+),?", r["Fave Sports"]), axis=1)


x["efs"] = x[["Fave Sports"]].apply(lambda r: , axis=1)


x[["Fave Sports", "efs"]]


split_sports = lambda s: (re.split(r"\)\s*,\s*|\s*,\s*", s["Fave Sports"]))

set(map(lambda s: s.strip().encode('ascii', errors="ignore"), pd.core.common.flatten(df[["Fave Sports"]].sample(100).apply(split_sports, axis=1).to_list())))







