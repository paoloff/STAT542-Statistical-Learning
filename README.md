# Project 1: Predicting the Housing Prices in Ames
# STAT 542 Statistical Learning - Fall 2022

Team:
Paolo Ferrari - paolof2 
Shitao Liu - sl53 
Yue Zhang - yuez11 

# I. Introduction

In this project, we predicted the price of houses in Ames from a large dataset categorized by location, size, year built, etc.

    - We utilized Linear Regression (LR) and Regression Tree (RT) models for prediction and refined each model with different regularization and feature engineering steps.

    - Our final models were LR with elastic net regularization and RT using XG Boost.

    - In the final 10x validation test, the root of mean squared error was 12.1 % for LR and 12.0 % for RT.

# II. Linear Regression Model

For the regression model, we did several pre-processing steps to boost the performance.

1. Feature Engineering

    New Features and Removed Features
        To enhance the explainability of the features used in the model, we added several new features and removed some.

        First of all, we created two features named “Remodelled” and “Age”, indicating whether the observation was re- modelled and its house age when sold. Another two features we added are “Sqft” and “Bathrooms”. “Sqft” is the sum of the “Gr_Liv_Area” and “Total_Bsmt_SF”, and “Bathrooms” is the total number of bathrooms of an observation, calculated as the weighted sum of the number of full baths and the number of half baths.

        We also removed several features, including “PID”, “Year_Sold”, “Mo_Sold”, “Year_Remod_Add”, “Year_Built”, “Garage_Yr_Blt”, “Ful_Bath”, “Half_Bath”, “Bsmt_Full_Bath”, “Latitude”, “Longitude”, “Condition_2”, “Heating”, “Pool_QC”, “Roof_Matl”, “Street”, “Utilities”, “Low_Qual_Fin_SF”, “Misc_Fea- ture”, and “Pool_Area”.

    Categorical Feature Encoding
        To quantify the categorical features, we did ordinal encoding first for the ordinal features and then one-hot encoding for nominal features, and the type of each categorical feature was decided based on the feature description.

        For ordinal encoding, the encoded value was based on the ordinality, ie, Poor = 1, Fair = 2, Average = 3... Particularly, we also encoded “Neighbourhood” as an ordinal feature. The ordinality of neighbourhoods was assigned based on the median housing sale price for these neighbourhoods in the training data.

        For one-hot encoding, each nominal feature is extended to several features based on the observed values in the training data. Values for a nominal fea- ture only seen in the testing data will not be encoded.

    Winsorization
        We did winsorization for several features, including “Lot_Frontage”, “Lot_Area”, “Mas_Vnr_Area”, “BsmtFin_SF_2”, “Bsmt_Unf_SF”, “Total_Bsmt_SF”, “Sqft”, “Second_Flr_SF”, “First_Flr_SF”, “Gr_Liv_Area”, “Garage_Area”, “Wood_Deck_SF”, “Open_Porch_SF”, “Enclosed_Porch”, “Three_season_porch”, “Screen_Porch”, and “Misc_Val”.

        We did winsorization for these features because they only impact the housing price linearly within a range. For example, when a house becomes surprisingly large, its indoor area no longer contributes to its price linearly. We set 95% as the upper bound, ie, values in both training and testing data larger than the 95 percentile value in the training data were replaced by it.

2. Model Selection

    We chose to use ElasticNet model in our implementation. The lambda range was np.exp(np.linspace(-1, -21, num=200)), and the alpha was 0.2. The model implementation was based on glmnet_python package.

3. Performance and Running Time

    Using ElasticNet with lambda range np.exp(np.linspace(-1, -21, num=200)) and alpha=0.2, we recorded error below benchmark for every train-test split.\


# III. Regression Tree Model

    For the Regression Tree Model, we used the XG Boost algorithm. Compared to the Linear Regression, this model required less pre-processing steps and gave very similar error rates.

    1. Data Pre-processing

        We utilized dummy coding and kept all features from train dataset. Keep all levels even if binary, and throw away all unseen features in the test dataset.

    2. Model Selection

        We utilized XGBoost with 1000 trees, 0.05 learning rate, each round use 50% randomly picked data to train the new tree. We used 100% of features when training a new tree.


# IV. Results
    We performed all tests on free Goggle's Colab.

    The results of the 10 test/train split results are the following:

    |ElasticNet Error| XGBoost Error | ElasticNet Time| XGBoost Time |
    |     0.1193     |     0.1129    |    6.916 sec   |   41.185 sec |
    |     0.1214     |     0.1178    |    6.630 sec   |   40.098 sec |
    |     0.1134     |     0.1117    |    7.152 sec   |   40.904 sec |
    |     0.1246     |     0.1145    |    7.797 sec   |   40.032 sec |
    |     0.1133     |     0.1112    |    6.897 sec   |   40.703 sec |
    |     0.1278     |     0.1296    |    6.857 sec   |   39.830 sec |
    |     0.1284     |     0.1227    |    6.649 sec   |   40.276 sec |
    |     0.1215     |     0.1253    |    7.081 sec   |   39.827 sec |
    |     0.1252     |     0.1243    |    7.751 sec   |   40.535 sec |
    |     0.1206     |     0.1272    |    10.087 sec  |   39.946 sec |

    The mean errors and execution times are shown below.

    | Mean ElasticNet Error | Mean XGBoost Error | Mean ElasticNet Time | Mean XGBoost Time |
    |          0.1215       |        0.1197      |       7.38 sec       |     40.33 sec     |
