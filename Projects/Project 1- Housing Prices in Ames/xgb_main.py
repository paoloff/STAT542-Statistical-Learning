import xgboost as xgb
import numpy as np
import pandas as pd

###########################################
# XGBoost main:

def boost_tree_pred(train, train_y, test, tree_n = 1000, eta=0.05, s_sample=0.5, colsample=1):
    #define the model

    xgb_r = xgb.XGBRegressor(n_estimators = tree_n, 
                             max_depth=6, learning_rate=eta, subsample=s_sample,
                             colsample_bytree =colsample)
   

    # Fitting the model
    xgb_r.fit(train, train_y)  
    # Predict the model
    pred = xgb_r.predict(test)

    return pred

def write_output(pred_output, test_raw):
    PID_output = test_raw['PID'].to_numpy()
    xgb_output = pd.DataFrame(np.vstack((PID_output, pred_output)).T, columns = ['PID','Sale_Price'])
    xgb_output = xgb_output.astype({'PID': 'int32'}).astype({'PID': 'str'})
    xgb_output = np.vstack((xgb_output.keys().to_numpy(),xgb_output.to_numpy()))
    np.savetxt('mysubmission.txt', xgb_output, delimiter=', ', fmt = '%s')
