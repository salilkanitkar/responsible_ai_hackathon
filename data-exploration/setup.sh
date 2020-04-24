pip install -q -U jupyter autoviz xlrd statsmodels xgboost autovizwidget pyarrow openpyxl pip bokeh==1.4.* panel==0.7.0 datashader pandas holoviews[recommended]==1.12.7 fastprogress
pip install -U -q tensorflow apache-beam==2.19 python-snappy pyarrow==0.15.* tensorflow_data_validation tensorflow-transform tensorflow-model-analysis \
    && jupyter nbextension enable --py widgetsnbextension \
    && jupyter nbextension enable --py tensorflow_model_analysis \
    && jupyter nbextension install --py --symlink tensorflow_model_analysis
pip install -U -q apache-beam[interactive]