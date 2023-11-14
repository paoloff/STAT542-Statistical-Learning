# Project 2: Predict Walmart Stores’ Weekly Sales Data
# STAT 542 Statistical Learning - Fall 2022

Team:
Paolo Ferrari - paolof2; 
Shitao Liu - sl53; 
Yue Zhang - yuez11;



## 1. Overview
In this project, we predicted the weekly sales data of Walmart stores with regard to different
stores and departments. By running gradient boosted regression tree method, we predict the
future two months (8 weeks)’ sales data: with more weight on holiday sales, we achieved a
weighted average absolute error (WAE) of 1599. To achieve this performance, we implemented
several strategies including data denoising and fitting different regression models on individual
store and department combination.

## 2. Data cleaning
The input training data is Walmart stores’ sales data from February 2010 to February 2011, with
regard to 45 different stores and 98 departments. The input training data has several problems.
First, not all weekly sales data is recorded, for specific stores and departments, the sales data is
missing. Second, the several departments’ sales data is completely missing. Third, for several
particular store and departments, there are outlier data points: those departments and stores
report negative sales. From a business standpoint, we treat this as some unexpected events
beyond normal operations. We therefore did the following:

  - For each store, department combination, we will add sales data if the record is missing
from the original training data. When filling these values, we can use either the average
department sales, scaled by the ratio of the store’s average sale over the average of all
45 stores’ sales data. Another choice is we simply refill every data point with 0. The
former can be justified by the view that those stores actually have these departments but
they fail to report the data. The later one is a reflection of speculation that some
departments may not exist in particular stores: for example, we can expect a department
of ‘cheese’ showing up only in stores in Wisconsin, while all other stores in the nation will
just sell cheese in the ‘diary’ department. Therefore we should simply fill the missing
value with 0. After testing, the performance supports the latter.

  - There may be an inherent, latent pattern for each department over time regardless of the
store, for example, we can expect the sales of sweets and chocolates will hike around
Valentine’s day and around winter holidays. Also, negative sales data may result from
some unexpected event, such as unrest, extreme weathers, etc. They are not
predictable, periodic events, so those data will lower the overall prediction strength. To
eliminate these outliers, we use SVD to filter the data and only keep top 8 features, and
reconstruct new training data from truncated SVD matrix. Here, the data we are filtering
are only from the training set: we performed SVD on over 4,000 data matrices
(store-dept combo).

## 3. Data encoding

  - For the raw data, we have the sales data as a function of store, department, and date. The
‘IsHoliday’ variable is ignored. For date, we first transform it into two variables: ‘Year’ and
‘Week’, where ‘Year’ may be 2010, 2011, 2012, and ‘Week’ may vary from 1 to 52. Due the
algorithm in python, we found out that all holidays are assigned to one specific week, for
example, Christmas is always assigned to Week 52. For the store and the department, we have
two encoding schemes:
  - We can do dummy coding for each store, each department, which will result in 4410
dummy variables. One obvious drawback is 4410 variables may be too many and will
significantly influence the performance negatively.
  - Or, we can run a regression tree model on each specific ‘Store’ and ‘Department’.

## 4. Model fitting

For fitting the models, we use gradient boosted regression trees. Compared with regular linear
regression, gradient boosted trees have more hyper parameters, while being essentially a
regression model itself. Therefore, by tuning the learning rate, the sub-sampling ratio, the tree
depth and the number of trees fitted, we can do fine tuning of our model.
When running the model, we find running the regression tree on over 4450 variables is
unsurprisingly slow, and it gives a mediocre WAE performance of 1900. We then try to run a
gradient boosted regression tree on each store and each department. This doesn’t reduce the
training time significantly, and actually due to the parallel nature of the former method, the latter
strategy cannot be accelerated even using a powerful GPU. For parameters, we use 50 tress for
fitting, with a learning rate of 0.3. We set the max depth to be 6 levels, and for each tree, we
sample 70% of all data to avoid over fitting.

## 5. Results

We use the Google Colab standard machine (not paid version), the model takes 19 minutes to
train.

- Performance: Here we list the performance for 10 test data sets. The average is 1566.
  
|Test set | Performance|
|--|--|
|1|1731.445|
|2|1434.749|
|3|1348.845|
|4|1391.556|
|5|2637.722|
|6|1653.632|
|7|1639.885|
|8|1344.911|
|9|1253.056|
|10|1228.402|
