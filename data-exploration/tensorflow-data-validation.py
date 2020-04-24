#!/usr/bin/env python
# coding: utf-8

get_ipython().system(' bash setup.sh')


from pathlib import Path
import tensorflow_data_validation as tfdv

print(f'TFDV version: {tfdv.version.__version__}')


data_folder = Path("../dataset")
# below paths should be realtive to data_folder
users_file_glob = "AllUsers.csv" 
ads_file_glob = "AllAds.csv"


users_stats = tfdv.generate_statistics_from_csv((data_folder/f"*{users_file_glob}").as_posix())


tfdv.visualize_statistics(users_stats)


user_schema = tfdv.infer_schema(statistics=users_stats)
tfdv.display_schema(schema=user_schema)


ads_stats = tfdv.generate_statistics_from_csv((data_folder/f"*{ads_file_glob}").as_posix())


tfdv.visualize_statistics(ads_stats)


ads_schema = tfdv.infer_schema(statistics=ads_stats)
tfdv.display_schema(schema=ads_schema)





import json


ad_data_path = "../dataset/Ads_GoogleVision_Annotated/1/1_annotations.json"

with open(ad_data_path) as f:
    # ad_features = json.load(f)
    print(f.read())


type(ad_features)




