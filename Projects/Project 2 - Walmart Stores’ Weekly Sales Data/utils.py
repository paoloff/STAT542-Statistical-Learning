# import basic libraries

from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder


def convert(x, drop=True):

    x_edited = x.drop(['Date'], axis=1)
    x_edited['Year'] = x['Date'].dt.isocalendar().year
    x_edited['Week'] = x['Date'].dt.isocalendar().week

    if drop:
        return x_edited[['Store','Dept', 'Year', 'Week', 'IsHoliday']]
    else:
        return x_edited[['Store','Dept', 'Year', 'Week', 'Weekly_Sales']]
    

def data_update(train, test, new_test = None, t=0):
    #append new_test to train
    if type(new_test) != None:
        train = pd.concat([train, new_test], ignore_index=True)

    start_date = pd.to_datetime('2011-03-01') + relativedelta(months = 2*t)
    end_date = pd.to_datetime('2011-03-01') + relativedelta(months = 2*(t+1))

    test_cropped = test[test['Date'] >= start_date]
    test_cropped = test_cropped[test_cropped['Date'] < end_date]

    return train, test_cropped.reset_index(drop=True)


def dummy_coding(train, test):

    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore', drop='if_binary')
    enc = ohe.fit_transform(train[['Week', 'IsHoliday']])

    # start transform
    train_dummy = pd.DataFrame(enc, columns=ohe.get_feature_names_out())
    test_dummy = pd.DataFrame(ohe.transform(test[['Week', 'IsHoliday']]), 
                            columns=ohe.get_feature_names_out())
    
    train_dummy['Year'] = train['Year'] - 2010
    train_dummy['Store'] = train['Store']
    train_dummy['Dept'] = train['Dept']

    test_dummy['Year'] = test['Year'] - 2010
    test_dummy['Store'] = test['Store']
    test_dummy['Dept'] = test['Dept']

    new_col_list = ohe.get_feature_names_out()
    new_col_list = ['Store', 'Dept', 'Year'] + new_col_list.tolist()

    
    return train_dummy[new_col_list], test_dummy[new_col_list]

def splitdata(x, list=['Store','Dept']):
    
    x_gb = x.groupby(list)
    x_ind = test_ind_1 = np.array([(x_gb.get_group(i))[list].iloc[0].to_list() for i in x_gb.groups])
    x_splitted = [(x_gb.get_group(i)).drop(list, axis=1) for i in x_gb.groups]

    return x_ind, x_splitted

def wmae(pred, new_test):
    weight = np.where(new_test['IsHoliday'].to_numpy(),5 ,1)
    test_y = new_test['Weekly_Sales'].to_numpy()
    wmae = np.sum(np.multiply(weight, np.abs(pred - test_y)))/np.sum(weight)

    return wmae

def denoise(input, magic_num=100, d=8):

    converted = convert(input, False)
    converted['Week_norm'] = converted['Year'] * magic_num + converted['Week']
    converted = converted.drop('Week', axis = 1)
    inds, converted_splitted = splitdata(converted, ['Dept'])
    inds = np.array(inds).reshape(-1)

    cropped_df = []

    ###iteration starts
    for i in range(inds.size):

        X = (converted_splitted[i]).drop(['Year'], axis=1).pivot(
            index='Store', columns='Week_norm', values='Weekly_Sales').fillna(0)
        X_mean = X.mean(axis=1).to_numpy()

        X_np = X.to_numpy()
        u, s, vh = np.linalg.svd((X_np.T - X_mean).T, full_matrices=False)
        s_cropped = np.diag(np.array([s[i] if i < d else 0 for i in range(s.size)]))

        X_pca = ((u @ s_cropped @ vh).T + X_mean).T

        #put cropped X back to df
        X_cropped = pd.DataFrame(X_pca, index=X.index, columns=X.columns)

        X_cropped = X_cropped.reset_index()
        X1 = pd.melt(X_cropped, id_vars='Store', value_vars=X_cropped.columns).sort_values(by=['Store','Week_norm'])
        final_df = pd.DataFrame([])

        final_df['Store'] = X1['Store']
        final_df['Year'] = np.floor(X1['Week_norm'].to_numpy().astype(int)/magic_num).astype(int)
        final_df['Week'] = np.mod(X1['Week_norm'].to_numpy().astype(int), magic_num)
        final_df['Weekly_Sales'] = pd.melt(X_cropped, id_vars='Store', value_vars=X_cropped.columns)['value']

        final_df = final_df.reindex()

        cropped_df.append(final_df)

    for i in range(inds.size):
        set_size = cropped_df[i].shape[0]
        cropped_df[i]['Dept'] = (np.ones(set_size)*inds[i]).astype(int)


    output = pd.concat(cropped_df).reset_index()

    holiday = [36, 47, 52, 6]
    wk = output['Week']
    output['IsHoliday'] = np.where((wk.isin(holiday)).to_numpy(),True,False)

    return output[['Store','Dept','Year','Week','IsHoliday','Weekly_Sales']]

def mypredict(train_data, test_data, new_test=None, t_input=1):

    t_test = t_input
    # mypredict happens

    # t for python starts from 0
    t = t_input - 1


    # update
    train_data, test_data = data_update(train_data, test_data, new_test, t)

    # prepare for output
    test_data_output = test_data.copy()
    train_update = train_data.copy()

    if t != t_test-1:
        return train_update, test_data_output

    if t == t_test-1:

        train_data = denoise(train_data)
        test_data = convert(test_data)

        train_y_ind, train_y_splitted = splitdata(train_data[['Store', 'Dept', 'Weekly_Sales']])
        
        #train_data = denoise(train_data)
        #test_data = convert(test_data)
        
        # next, do dummy coding 
        train_data, test_data = dummy_coding(train_data, test_data)


        # next, generate ind and values for train, test data
        train_ind, train_splitted = splitdata(train_data)
        test_ind, test_splitted = splitdata(test_data)

        magic_num = 10000

        #squeeze ind set into 1D
        train_ind_1d = train_ind @ np.array([magic_num,1])
        test_ind_1d = test_ind @ np.array([magic_num,1])




        pred_final = []
        # only do linReg when test needs
        for i, ind in enumerate(test_ind_1d):
            if not ind in train_ind_1d:
                test_temp = test_splitted[i]
                #print(ind,', Oh! I cannot find a model:(')
                pred = np.zeros(test_temp.shape[0])
            else:
                # find corresponding {store, dept}
                loc = np.argwhere(train_ind_1d == ind)[0,0]
                #load data
                train_temp = train_splitted[loc]
                train_y_temp = train_y_splitted[loc]
                test_temp = test_splitted[i]

                #linReg = LinearRegression()
                #linReg.fit(train_temp, train_y_temp)
                #pred = linReg.predict(test_temp)

                xgb_r = xgb.XGBRegressor(n_estimators = 50, 
                                            max_depth=6, learning_rate=0.3, 
                                            colsample_bytree=0.7,
                                            subsample=1.0,
                                        verbosity=0
                                        )
                # Fit the model
                xgb_r.fit(train_temp.to_numpy().astype(float), train_y_temp.to_numpy().astype(float))  
                # Predict the model
                pred = xgb_r.predict(test_temp.to_numpy().astype(float))

                pred = pred.ravel()

                ########################################
                # special treatment for fold 5

                week_48_ind = -1
                week_52_ind = -1

                for k in range(pred.size):
                    if test_temp['Week_48'].iloc[k] == 1:
                        week_48_ind = k
                    if test_temp['Week_52'].iloc[k] == 1:
                        week_52_ind = k
                
                if week_48_ind != -1 and week_52_ind != -1:
                    #pred[week_48_ind] += pred[week_48_ind] /10
                    #pred[week_52_ind] = 2* week_52_sales
                    pass
                # special treatment ends
                




            pred_final += pred.ravel().tolist()
                
                



        pred_final = np.array(pred_final)
        test_data_output['Weekly_Pred'] = pred_final

        

        #verify = pd.read_csv('fold_1.csv', parse_dates=['Date'])
        #wmae(pred_final, verify)
        return train_update, test_data_output

