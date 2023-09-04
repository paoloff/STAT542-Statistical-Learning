import numpy as np
from scipy import stats
import pandas as pd
import glmnet_python
from glmnet import glmnet
from cvglmnet import cvglmnet
from cvglmnetPredict import cvglmnetPredict

import elastic_preprocess


###########################################
# ElasticNet main

def predict(train, test):
    train, _ = elastic_preprocess.processing(train)
    test, pid = elastic_preprocess.processing(test)
    rating = elastic_preprocess.neighb_rate(train)
    
    train_X = train.drop(['Sale_Price'], axis=1)
    train_Y = np.log(train[['Sale_Price']]).to_numpy(dtype=np.float64)

    test_X = test

    wins_feature = ["Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", 
                    "Bsmt_Unf_SF", "Total_Bsmt_SF", 'Sqft', "Second_Flr_SF", 
                    'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", 
                    "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val"]

    for col in wins_feature:
        # if (col in ['Second_Flr_SF', 'First_Flr_SF']):
        train_X[col] = stats.mstats.winsorize(train_X[col], limits=[0, 0.05], inplace=True)
        # else:
          # train_X[col] = sp.stats.mstats.winsorize(train_X[col], limits=[0, 0.03], inplace=True)
        max_thresh = train_X[col].max()
        for i in range(test_X[col].shape[0]):
            if (test_X[col].iloc[i] > max_thresh):
                test_X[col].iloc[i] = max_thresh

    ## List all ordinal features
    ordinal_features = ['Lot_Shape', 'Land_Slope', 
                        'Overall_Qual', 'Overall_Cond', 'Exter_Qual', 'Exter_Cond', 
                        'Bsmt_Qual', 'Bsmt_Cond', 'Bsmt_Exposure', 'BsmtFin_Type_1', 'BsmtFin_Type_2', 
                        'Heating_QC', 'Electrical', 'Kitchen_Qual', 'Functional', 'Fireplace_Qu', 
                        'Garage_Finish', 'Garage_Qual', 'Garage_Cond', 'Paved_Drive', 
                        'Fence', 'Neighborhood']

    ## Preprocess training and testing data separately
    encoded_train, cate_features, cate_values = elastic_preprocess.mix_encoder(train_X, ordinal_features, rating)

    ## Use the categorical features and values from training set to 
    ## do one-hot encoding for the testing set
    ord_encoded_test = elastic_preprocess._ord_encode(test_X, ordinal_features, rating)
    encoded_test = elastic_preprocess._encode(ord_encoded_test, cate_features, cate_values)

    encoded_train_X = encoded_train.to_numpy()
    encoded_test_X = encoded_test.to_numpy()

    elastic_seq = np.exp(np.linspace(-1, -21, num=200))
    elastic = cvglmnet(x = encoded_train_X.copy(), y = train_Y.copy(), alpha = 0.2, lambdau = elastic_seq)
    elastic_pred = cvglmnetPredict(elastic, s = elastic['lambda_min'], newx=encoded_test_X.copy())
    
    return elastic_pred, pid

###########################################
# ElasticNet output
def write_output(pred, pid):
    pred = np.exp(pred).squeeze()
    pidd = pid.to_numpy().squeeze()
    pd_output = pd.DataFrame(np.vstack((pidd, pred)).T, columns = ['PID','Sale_Price'])
    pd_output = pd_output.astype({'PID': 'int32'}).astype({'PID': 'str'})
    np_output = np.vstack((pd_output.keys().to_numpy(),pd_output.to_numpy()))
    np.savetxt('mysubmission1.txt', np_output, delimiter=', ', fmt = '%s')