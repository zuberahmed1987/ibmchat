#!/bin/bash

export DIRNAME=$(dirname $0)

cd ${DIRNAME}

. ../env/bin/activate

exec streamlit run ChatUI.py --browser.gatherUsageStats=False
