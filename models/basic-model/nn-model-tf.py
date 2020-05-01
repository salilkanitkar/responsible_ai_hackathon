#!/usr/bin/env python
# coding: utf-8

get_ipython().system(' ./setup.sh')


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


RANDOM_SEED = int(time.time())

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"Using random seed, {RANDOM_SEED}")


DATA_FOLDER = Path("../../dataset/")
BATCH_SIZE = 4096 # bigger the batch, faster the training but bigger the RAM needed
TARGET_COL = "Rating"

# data files path are relative DATA_FOLDER
users_ads_rating_csv = DATA_FOLDER/"users-ads-without-gcp-ratings_OHE_MLB.csv"


pd.read_csv(users_ads_rating_csv).columns


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
RATING = "Rating"
AD_NUM_FACES = "ad_num_faces"



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
    RATING: "**",
    AD_NUM_FACES: "**"
}

# SELECTED_COLS = [AGE, ZIP_CODE, FAVE_SPORTS, GENDER, HOME_COUNTRY, HOME_TOWN, INCOME, MOST_LISTENED_MUSICS, MOST_READ_BOOKS, 
#                  MOST_VISITED_WEBSITES, MOST_WATCHED_MOVIES, MOST_WATCHED_TV_PROGRAMMES, TIMEPASS, TYPE_OF_JOB, WEEKLY_WORKING_HOURS, 
#                  FAVE1, FAVE2, FAVE3, FAVE4, FAVE5, FAVE6, FAVE7, FAVE8, FAVE9, FAVE10, UNFAVE1, UNFAVE2, UNFAVE3, UNFAVE4, UNFAVE5, 
#                  UNFAVE6, RATING]

AD_FACE_COLS = [AD_NUM_FACES]
AD_LABEL_COLS = []
AD_OBJECT_COLS = []
AD_SAFE_SEARCH_COLS = []


SELECTED_AD_COLS = AD_FACE_COLS + AD_LABEL_COLS + AD_OBJECT_COLS + AD_SAFE_SEARCH_COLS


SELECTED_INP_COLS = [AGE, ZIP_CODE, FAVE_SPORTS, GENDER_F, GENDER_M] + SELECTED_AD_COLS
SELECTED_COLS = SELECTED_INP_COLS + [TARGET_COL]


SELECTED_COLS


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


def ad_dataset_pd():
    return pd.read_csv(users_ads_rating_csv, usecols=SELECTED_COLS, dtype=str)


ad_dataset_pd().sample(3)


def dict_project(d:Dict, cols:List[str]) -> Dict:
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

def fix_age_tf(example:Dict, new_col_suffix=""):
    """Wrap in a py_function for TF to run inside its execution graph"""
#     example[AGE + new_col_suffix] = tf.py_function(fix_age, [example[AGE]], (tf.float32, ))
    example[AGE + new_col_suffix] = tf.py_function(fix_age, [example[AGE]], tf.float32)
    example[AGE + new_col_suffix] = tf.expand_dims(example[AGE + new_col_suffix], 0) # https://github.com/tensorflow/tensorflow/issues/24520#issuecomment-579421744
    return example


fix_age("50"), fix_age("50.5"), fix_age("-10"), fix_age("bad_age_10"), fix_age("300")


fix_age_tf_fn = partial(fix_age_tf, new_col_suffix="_encoded")
for d in ad_dataset(1, True).map(fix_age_tf_fn).batch(3).take(5):
    pprint(dict_project(d, [AGE, AGE + "_encoded"]))
    print()


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

def fix_zip_code_tf(example:Dict, n_digits=FIRST_K_ZIP_DIGITS, indexer=zip_code_indexer, new_col_suffix=""):
    """Creates new columns for the first n_digits in zip_code"""
    fix_zip_code_fn = partial(fix_zip_code, n_digits=n_digits, indexer=indexer)
    example[ZIP_CODE + new_col_suffix] = tf.py_function(fix_zip_code_fn, [example[ZIP_CODE]], tf.float32)
    example[ZIP_CODE + new_col_suffix].set_shape(len(indexer) * n_digits)
    return example


test_zip_code_indexer = IndexerForVocab(string.digits)

(fix_zip_code("43556", 10, test_zip_code_indexer),
fix_zip_code("43556", 2, test_zip_code_indexer),
fix_zip_code("43556", 4, test_zip_code_indexer),
fix_zip_code(None, 3, test_zip_code_indexer))


test_zip_code_indexer = IndexerForVocab(string.digits)

(fix_zip_code(tf.constant([b"43556"], shape=(1,), dtype=tf.string), 10, test_zip_code_indexer),
fix_zip_code(tf.constant([b"43556"], shape=(1,), dtype=tf.string), 2, test_zip_code_indexer),
fix_zip_code(tf.constant([b"43556"], shape=(1,), dtype=tf.string), 4, test_zip_code_indexer),
fix_zip_code(tf.constant([43556], shape=(1,), dtype=tf.int32), 4, test_zip_code_indexer),\
fix_zip_code(None, 3, test_zip_code_indexer))


fix_zip_code_tf_fn = partial(fix_zip_code_tf, new_col_suffix="_encoded")
for d in ad_dataset(1, True).map(fix_zip_code_tf_fn).batch(5).take(3):
    pprint(dict_project(d, [ZIP_CODE, ZIP_CODE + "_encoded"]))
    print()


FAV_SPORTS_UNKNOWN = "UNK_SPORT"
ALL_FAV_SPORTS = ['Olympic sports', 'Winter sports', 'Nothing', 'I do not like Sports', 'Equestrian sports', 'Skating sports', 'Precision sports', 'Hunting sports', 'Motor sports', 'Team sports', 'Individual sports', 'Other', 'Water sports', 'Indoor sports', 'Endurance sports']

fav_sports_binarizer = MultiLabelBinarizer()
fav_sports_binarizer.fit([ALL_FAV_SPORTS])


# Attempt to write purely in TF graph
# def fix_fav_sports(sports_str:str, topk=2, pad_constant="PAD_SPORT") -> List[str]:
#     sports = tf.strings.regex_replace(sports_str, r"\s*\(.*,.*\)\s*", "")
#     sports = tf.strings.regex_replace(sports, r"\s*,\s*", ",")
#     sports = tf.strings.split(sports, ",").numpy()[:topk]
#     tf.print(sports.shape[0])
#     right_pad_width = max(0, topk - sports.shape[0])
#     result = np.pad(sports, (0, right_pad_width), constant_values=pad_constant) 
#     return result


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

def fix_fav_sports_tf(example:Dict, first_k=2, pad_constant="PAD_SPORT", new_col_suffix:str=""):
    """Existing column will not be overriden with new_col_suffix"""
    example[FAVE_SPORTS + new_col_suffix] = tf.py_function(fix_fav_sports_mlb, [example[FAVE_SPORTS]], tf.float32)
    example[FAVE_SPORTS + new_col_suffix].set_shape(len(ALL_FAV_SPORTS))
    return example


(
    fix_fav_sports_mlb(tf.constant([b"Individual sports (Tennis, Archery, ...), Indoor sports, Endurance sports, Skating sports"])),
    fix_fav_sports_mlb(tf.constant([b"Skating sports"])),
    fix_fav_sports_mlb(tf.constant([b"Individual sports (Tennis, Archery, ...)"])),
    fix_fav_sports_mlb(tf.constant([b"Indoor sports, Endurance sports, Skating sports"])),
)


(
    fix_fav_sports_mlb("Individual sports (Tennis, Archery, ...), Indoor sports, Endurance sports, Skating sports"),
    fix_fav_sports_mlb("Skating sports"),
    fix_fav_sports_mlb("Individual sports (Tennis, Archery, ...)"),
    fix_fav_sports_mlb("Indoor sports, Endurance sports, Skating sports"),
)


fix_fav_sports_tf_fn = partial(fix_fav_sports_tf, new_col_suffix="_new")
for d in ad_dataset(1, True).map(fix_fav_sports_tf_fn).batch(5).take(2):
    pprint(dict_project(d, [FAVE_SPORTS, FAVE_SPORTS + "_new"]))
    print()


RATINGS_CARDINALITY = 5 # not zero based indexing i.e. ratings range from 1 to 5


def create_target(example:Dict):
    y = tf.one_hot(
        tf.cast(tf.strings.to_number(example[RATING], tf.float32), tf.int32), 
        RATINGS_CARDINALITY)
    example.pop(RATING)
    
    return example, y


def create_target_pd(rating_str:str):
    return np.eye(RATINGS_CARDINALITY, dtype=int)[int(float(rating_str)) - 1]


def create_dataset_tf() -> tf.data.Dataset:
    return ad_dataset(1, True).        map(fix_age_tf, tf.data.experimental.AUTOTUNE).        map(fix_zip_code_tf, tf.data.experimental.AUTOTUNE).        map(fix_fav_sports_tf, tf.data.experimental.AUTOTUNE).        map(create_target, tf.data.experimental.AUTOTUNE)


# Credits: https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough?hl=en#create_a_tfdatadataset
def pack_features_vector(features:Dict, labels, cols:List[str]=[AGE]):
    """Pack the features into a single array for the list of cols"""
    # features = tf.stack(list(dict_project(features, cols).values()), axis=1)
    features = tf.concat(list(dict_project(features, cols).values()), axis=1)
    return features, labels


for d in create_dataset_tf().batch(2).map(pack_features_vector).take(2):
    pprint(d)


def transform_pd_X(df:pd.DataFrame, inp_cols:List[str]):
    """Original dataframe will be modified"""
    df[AGE] = df[AGE].apply(lambda age: [fix_age(age)])
    df[ZIP_CODE] = df[ZIP_CODE].apply(lambda zc: fix_zip_code(zc, n_digits=2, indexer=zip_code_indexer))
    df[FAVE_SPORTS] = df[FAVE_SPORTS].apply(fix_fav_sports_mlb)
    df[GENDER_F] = df[GENDER_F].apply(lambda gender_f: [int(gender_f)])
    df[GENDER_M] = df[GENDER_M].apply(lambda gender_m: [int(gender_m)])
    df[AD_NUM_FACES] = df[AD_NUM_FACES].apply(lambda ad_num_faces: [int(ad_num_faces)])
    df["X"] = df[inp_cols].apply(np.concatenate, axis=1)
    # TODO: vectorize, else inefficient to sequentially loop over all example
    X = np.array([x for x in df["X"]])
    return X


def transform_pd_y(df:pd.DataFrame, target_col:str):
    """Original dataframe will be modified"""
    df["y"] = df[target_col].apply(create_target_pd)
    # TODO: vectorize, else inefficient to sequentially loop over all example
    y = np.array([y for y in df["y"]])
    return y


def create_dataset_pd(inp_cols:List[str]=SELECTED_INP_COLS, target_col:str=TARGET_COL, fraction:float=1) -> pd.DataFrame:
    """Prepare the dataset for training on a fraction of all input data"""
    df = ad_dataset_pd().sample(frac=fraction)
    return transform_pd_X(df, inp_cols), transform_pd_y(df, target_col)


# Input builders
def input_fn_train(batch_size=10):
    return create_dataset.        shuffle(2 * batch_size).batch(batch_size, drop_remainder=True).        map(pack_features_vector, tf.data.experimental.AUTOTUNE).        cache().prefetch(tf.data.experimental.AUTOTUNE)

def input_fn_eval(batch_size=10, cache=True):
    # TODO: use dataset's skip & take to create train and validation datasets
    val_dataset = create_dataset(test_files).batch(batch_size)
    if cache: val_dataset = val_dataset.cache()
    return val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

def input_fn_predict():
    # return tf.data.Dataset.from_tensor_slices({"x": tf.cast(X_test, tf.int32)}).batch(1)
    pass


for d in input_fn_train(2).take(2):
    pprint(d)


from tensorboard import notebook


get_ipython().run_line_magic('reload_ext', 'tensorboard')


get_ipython().run_line_magic('tensorboard', '--logdir logs --port 6006')


notebook.list()


get_ipython().run_cell_magic('time', '', '\n# train_dataset = input_fn_train(BATCH_SIZE)\nX, y = create_dataset_pd()')


keras_model_metrics = [
    "accuracy",
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'), 
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]
train_histories = []


# DON'T CHANGE THE EPOCHS VALUE
BATCH_SIZE = 4096
EPOCHS = 1000


logdir = Path("logs")/datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    logdir, 
    histogram_freq=max(1, ceil(EPOCHS / 20)), # to control the amount of logging
#     embeddings_freq=epochs,
)
print(f"Logging tensorboard data at {logdir}")


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


get_ipython().run_cell_magic('time', '', '\ntrain_histories.append(model.fit(\n    X, y,\n    BATCH_SIZE,\n    epochs=EPOCHS, \n    callbacks=[tensorboard_callback, tfdocs.modeling.EpochDots()],\n    validation_split=0.2,\n    verbose=0\n))')


histories_dict = train_histories[-1].history
for metric in histories_dict.keys():
    print(metric, histories_dict[metric][-1])


model.save((logdir/"keras_saved_model").as_posix(), save_format="tf")


PredictionReport = namedtuple("PredictionReport", "probabilities predicted_rating confidence")

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

