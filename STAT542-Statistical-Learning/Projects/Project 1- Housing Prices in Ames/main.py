import numpy as np
import pandas as pd
import elastic_main
import xgb_preprocess
import xgb_main
from split_train_test import split

#### Splitting and savig data into separate train and test files
split("Ames_data.csv")

## Read train data
train_raw = pd.read_csv('./train.csv')

## Read test data:
test_raw = pd.read_csv('./test.csv')

## Run prediction on test data using ElasticNet:
pred_en, pid_en = elastic_main.predict(train_raw, test_raw)

## Run prediction on test data using XGBoost:
train, train_y, test = xgb_preprocess.get_data(train_raw, test_raw)
zero_cols = np.where(np.std(test, axis = 0)==0)[0]
pred_xgb = xgb_main.boost_tree_pred(train, np.log(train_y), test)
pred_output = np.exp(pred_xgb)

## Write output files
elastic_main.write_output(pred_en, pid_en)
xgb_main.write_output(pred_output, test_raw)
