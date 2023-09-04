from collections import defaultdict


###########################################
# ElasticNet preprocessing

def processing(ames_data):
    ames_data['Remodel'] = [int(ames_data.iloc[i]['Year_Built'] == ames_data.iloc[i]['Year_Remod_Add']) for i in range(ames_data.shape[0])]
    ames_data['Age'] = ames_data['Year_Sold'] - ames_data['Year_Built']
    ames_data['Sqft'] = ames_data['Gr_Liv_Area'] + ames_data['Total_Bsmt_SF']
    ames_data['Bathrooms'] = ames_data['Full_Bath'] + 0.5 * ames_data['Half_Bath'] + 0.75 * ames_data['Bsmt_Full_Bath'] + 0.25 * ames_data['Bsmt_Half_Bath']
    pid = ames_data[['PID']]
    ames_data = ames_data.drop(['PID',
                                'Year_Sold', 'Mo_Sold', 'Year_Remod_Add', 
                                'Year_Built', 'Garage_Yr_Blt',
                                'Full_Bath', 'Half_Bath', 
                                'Bsmt_Full_Bath', 'Bsmt_Half_Bath', 
                                'Latitude', 'Longitude',
                                'Condition_2', 'Heating', 'Pool_QC', 'Roof_Matl', 'Street', 'Utilities',
                                'Low_Qual_Fin_SF', 'Misc_Feature', 'Pool_Area'], axis=1)
    return ames_data, pid


"""## One-hot encoding"""

## This function returns all possible values for a given feature in key: val pair
## eg. all_values['Alley'] = [Gravel, Paved, No alley access]
def _get_categorical_values(data, features):
    all_values = {}
    for feature in features:
        vals = set()
        for sub in data[feature].values:
            vals.add(sub)
        all_values[feature] = list(vals)
    return all_values

## This function returns all categorical (non-numerical and non-binary)
## variables from the input dataset
def _get_categories(data):
    categorical_features = []
    for col in data.columns:
        if (data[col].describe().dtype == 'object'):
            categorical_features.append(col)
    return categorical_features

## This function does one-hot encoding for you
## Input: a dataframe, 
##        list of categorical features (can be generated from _get_categories)
##        key: val pair of categorical values (from _get_categorical_values())
## Output: a dataframe with all categorical variables encoded as one-hot
##        each value for each categorical variable will become a new variable 
##        in the new dataframe
def _encode(data, features, values):
    cate_sub = data[features]
    nume_sub = data.drop(features, axis=1)
    
    ## One-hot coding: each possible value for each feature becomes an independent col;
    for feature in features:
        types = cate_sub[feature]
        for value in values[feature]:
            vals = [int(types.iloc[i] == value) for i in range(types.shape[0])]
            nume_sub[feature + '_' + value] = vals

    ## Below is for asserting the correctness of one-hot coding
#     for feature in features:
#         for i in range(cate_sub.shape[0]):
#             typer = cate_sub.iloc[i][feature]
#             try:
#                 assert(nume_sub.iloc[i][feature + '_' + typer] == 1)
#             except:
#                 print(i, feature)
    return nume_sub

## One-hot Encoder
## Input: dataframe
## Output: one-hot encoded dataframe, 
##        categorical features, categorical values (to encode the testset)
def categorical_encoder(data):
    categorical_features = _get_categories(data)
    categorical_values = _get_categorical_values(data, categorical_features)
    encoded_data = _encode(data, categorical_features, categorical_values)
    return encoded_data, categorical_features, categorical_values


"""## One-hot + ordinal encoding"""

## The ordinal encoder: encode all specified ordinal features
## Input: dataframe, categorical features
## Output: dataframe with categorical features encoded (no new features added)
def _ord_encode(data, features, rating):
    cate_sub = data[features]
    nume_sub = data.drop(features, axis=1)
    encoding = defaultdict(int)

    ## Change the encoding map to get the best performance
    for feature in features:
        if (feature in ['Overall_Qual', 'Overall_Cond']):
            encoding = {'Very_Poor': 1, 'Poor': 2, 'Fair': 3, 'Below_Average': 4,
                        'Average': 5, 'Above_Average': 6, 'Good': 7, 'Very_Good': 8,
                        'Excellent': 9, 'Very_Excellent': 10}

        elif (feature in ['Heating_QC', 'Kitchen_Qual', 'Exter_Qual', 'Exter_Cond']):
            encoding = {'Poor': 1, 'Fair': 2, 'Typical': 3, 'Average': 4, 'Good': 5, 'Excellent': 6}

        elif (feature in ['Bsmt_Qual', 'Bsmt_Cond']):
            encoding = {'No_Basement': 1, 'Poor': 2, 'Fair': 3, 'Typical': 4, 'Good': 5, 'Excellent': 6}
        
        elif (feature == 'Bsmt_Exposure'):
            encoding = {'No_Basement': 1, 'No': 2, 'Mn': 3, 'Av': 4, 'Gd': 5}
          
        elif (feature in ['BsmtFin_Type_1', 'BsmtFin_Type_2']):
            encoding = {'No_Basement': 1, 'Unf': 2, 'LwQ': 3, 'Rec': 4, 'BLQ': 5, 'ALQ': 6, 'GLQ': 7}

        elif (feature == 'Fireplace_Qu'):
            encoding = {'No_Fireplace': 1, 'Poor': 2, 'Fair': 3, 'Average': 4, 'Typical': 4, 'Good': 5, 'Excellent': 6}

        elif (feature in ['Garage_Qual', 'Garage_Cond']):
            encoding = {'No_Garage': 1, 'Poor': 2, 'Fair': 3, 'Average': 4, 'Typical': 4, 'Good': 5, 'Excellent': 6}

        elif (feature == 'Garage_Finish'):
            encoding = {'No_Garage': 1, 'Unf': 2, 'RFn': 3, 'Fin': 4}

        elif (feature == 'Pool_QC'):
            encoding = {'No_Pool': 1, 'Fair': 2, 'Average': 3, 'Typical': 3, 'Good': 4, 'Excellent': 5}

        elif (feature == 'Fence'):
            encoding = {'No_Fence': 1, 'Minimum_Wood_Wire': 2, 'Good_Wood': 3, 'Minimum_Privacy': 4, 'Good_Privacy': 5}

        elif (feature == 'Lot_Shape'):
            encoding = {'Irregular': 1, 'Moderately_Irregular': 2, 'Slightly_Irregular': 3, 'Regular': 4}
          
        elif (feature == 'Utilities'):
            encoding = {'NoSeWa': 1, 'NoSewr': 2, 'AllPub': 3}
          
        elif (feature == 'Land_Slope'):
            encoding = {'Sev': 1, 'Mod': 2, 'Gtl': 3}

        elif (feature == 'Electrical'):
            encoding = {'Mix': 1, 'FuseP': 2, 'FuseF': 3, 'FuseA': 4, 'SBrkr': 5, 'Unknown': 3}

        elif (feature == 'Functional'):
            encoding = {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}
          
        elif (feature == 'Paved_Drive'):
            encoding = {'Dirt_Gravel': 1, 'Partial_Pavement': 2, 'Paved': 3}

        elif (feature == 'Neighborhood'):
            encoding = rating

        types = cate_sub[feature]
        encoded = [0 for _ in range(types.shape[0])] 
        for i in range(types.shape[0]):
            encoded[i] = encoding[types.iloc[i]]
        nume_sub[feature] = encoded

    return nume_sub

## The mix_encoder function: do level-encoding for ordinal features first, 
## then one-hot encoding for the moninal features
def mix_encoder(data, ordinal_features, rating):
    ordinal_encoded = _ord_encode(data, ordinal_features, rating)
    remaining_features = _get_categories(ordinal_encoded)
    remaining_values = _get_categorical_values(ordinal_encoded, remaining_features)
    fully_encoded = _encode(ordinal_encoded, remaining_features, remaining_values)
    return fully_encoded, remaining_features, remaining_values

def neighb_rate(train):
    neighbourhood_rating = defaultdict(int)
    rating = 1
    for neighbourhood in train.groupby('Neighborhood').agg('median')['Sale_Price'].sort_values().index:
        neighbourhood_rating[neighbourhood] = rating
        rating += 2
    return neighbourhood_rating