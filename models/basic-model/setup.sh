#!/bin/bash 
conda install -y pandas

if [ -d AIF360 ] ; then
    echo "AI Fairness module from IBM already exists and will not be installed"
else
    git clone https://github.com/IBM/AIF360 && cd AIF360/ && pip install --editable '.[all]'
    echo "Installed AI Fairness module from IBM"
fi

pip install -U -q pip tensorflow==2.1 git+https://github.com/tensorflow/docs numpy chakin scikit-learn imblearn bokeh==1.4.* panel==0.7.0 holoviews[recommended]==1.12.7

echo "Restart kernel if any new modules are installed to be sure if the changes are picked up"