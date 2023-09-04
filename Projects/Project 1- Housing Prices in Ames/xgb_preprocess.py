import pandas as pd
from sklearn.preprocessing import OneHotEncoder

###########################################
# XGBoost preprocessing

def get_data(train_raw, test_raw):

    train_y_raw = train_raw['Sale_Price']
    train_raw = train_raw.drop(['PID','Sale_Price'], axis = 1)
    test_raw = test_raw.drop(['PID'], axis = 1)
    
    #first, fill all NaN values with 0
    train_raw = train_raw.fillna(0)
    test_raw = test_raw.fillna(0)


    #next, do dummy coding
    #first, pick out cat variables and numerical variables

    cat_var = []
    num_var = []

    for var in train_raw.keys().tolist():
        if train_raw[var].dtype == 'O':
            cat_var.append(var)
        else:
            num_var.append(var)


    #dummy coding train set
    #colinearity checked
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    enc = ohe.fit_transform(train_raw[cat_var])
    #Converting back to a dataframe 
    train_dummy = pd.DataFrame(enc, columns=ohe.get_feature_names_out())
    test_dummy = pd.DataFrame(ohe.transform(test_raw[cat_var]), columns=ohe.get_feature_names_out())

    train_encoded = pd.concat([train_dummy,train_raw[num_var]], axis = 1)
    test_encoded = pd.concat([test_dummy,test_raw[num_var]], axis = 1)

    train_np = train_encoded.to_numpy()
    test_np = test_encoded.to_numpy()

    train_y_np = train_y_raw.to_numpy()

    return train_np, train_y_np, test_np