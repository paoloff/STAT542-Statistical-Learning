# Project 3 - Movie Review Sentiment Analysis
# STAT 542 Statistical Learning - Fall 2022

Team:
Paolo Ferrari - paolof2; 
Shitao Liu - sl53; 
Yue Zhang - yuez11;


## 1. Introduction

-  Sentiment analysis is a tool for classifying the subjective impression of a product or topic.
-  Using sentiment analysis, an algorithm can read a text written in natural language and map it to a scale ranging from positive to negative feelings.
-  Because it is able to interpret human language, sentiment analysis is widely used in many online platforms.
-  For instance, companies use sentiment analysis to gain direct feedback of customers of a product.

## 2. Overview of the project

- In this project, we create a sentiment analysis algorithm to interpret movie reviews in the IMDB website.
- The data consists of 50000 reviews which have scores from 0 to 10.
- If the score is below or equal to 4, it is classified as negative. If the score is 7 or bigger, it is classified as positive (the reviews with scores of 5 and 6 are excluded from this analysis).
- The goal of the algorithm is to guess if the review is negative or positive directly from the text written by the reviewer.

## 3. Data processing
The data set is a table with 50000 rows and 4 columns consisting of
    i. Id, the identification number of each review
    ii. Sentiment, 0 for negative and 1 for positive
    iii. Score, from 0 to 10 and excluding 5 and 6.
    iv. Review, the actual text written by the reviewer.
The data processing consists of 2 main steps: (1) removing irrelevant symbols and (2) filtering the text using a vocabulary.

- 3.1. Bag-of-words encoding

  - A common class of models for sentiment analysis are the so-called bag-of-words models. In this type of model, the only elements that matter in a body of text are the words themselves, excluding punctuation and the order in which words apper. Therefore, the first thing we do in processing the data is to remove all punctuations marks and make all letters in miniscule form.
  - Besides making the data easier to process in subsequent steps, removing punctuation also removes any HTML tags attached to the text which could cause problems later on.

- 3.2. Filtering the text using a vocabulary
  
    - The second main step in processing the review texts is to remove most words and only use a small subset of them. The reason for this is that usually by only knowing a small subset of words in a review is enough to guess the impression of the reviewer, such as “best” and “great” for positive reviews and “worst” or “waste” for negative ones. By removing the unnecessary or less meaningful words, we can simplify the complexity of the problem and prevent overfitting of the model.
    - The vocabulary is found in R. It is worth mentioning here that we utilized the data from all reviews to obtain the vocabulary, not only the train data of a given split. Technically this would be a little of a cheat, but it was allowed by Prof. Liang since we have access to all data anyway.
    - The steps to obtain the vocabulary are:
          i. Construct a vectorized representation of all texts with sequence of words up to 4. Each term (sequence of 1 to 4 words) is a column in this representation, and each row is a review.
        ii. Make a Logistic Regression with Lasso with this data.
        iii. Select the model with largest degree of freedom less than 2000.
        iv. Make another Logistic Regression with Lasso with a restricted vocabulary obtained from the previous step.
        v. Select the model with the degree of freedom closest to 1000.
    - In this way, we obtain a vocabulary with 1005 terms in R.
    - Later, in Python, we reduce this to 980 terms by equating different words with the same letters (e.g. “its” and “it’s”). The 980 terms are submitted in a separate text file.
      
## 4. Model

- Our model consists of a neural network with 3 layers. It takes a vectorized form of a review as an input and outputs the probability that the review has a positive score. More specifically,
    - The first layer is the input layer with dimensionality 980. Each entry in this 980-dimensional vector is the number of times a term in the vocabulary appears in that specific review.
    - The second and hidden layer has a dimensionality of 20 (or, equivalently, this layer has N = 20 neurons). The activation in each neuron is given by the output of the previous layer. The reason we chose N = 20 was merely by trial and error. We found for N > 50, the neural network tend to overfit the data, and for N < 5, the performance gets affected. The activation function for the output of this layer is a ReLU function. We chose this due to the commonality and good nonlinearity provided by this function.
    - The third and last layer is a scalar (single neuron). The activation for the output is a sigmoid function, thus providing a real number between 0 and 1. This is the output of the neural network and is the probability that the review is a positive one.
    - We build and train the neural network using the Keras package from Tensorflow. The cost function is the binary cross entropy (commonly used for classification tasks) and the number of training epochs is 20. The number of epochs was also found by trial and error. Too few epochs and the neural network is under-optimized. Too many epochs and the neural network overfits the data.
    - The initialization of the Neural Network is the common Adam initialization.
    - Train and test splits are done 50/50 of the entire dataset, that is, 25000 reviews are used to train the neural network and 25,000 reviews are used to evaluate it.
    - There are a total of 5 splits, with random id’s given for the train and test reviews. The performance of the model is evaluated in each of these 5 splits.
      
## 5. Result
The model is evaluated by the common area-under-the-curve (AUC) of the test data. Below we provide the values for each split:

| Split number| AUC|
| --| --|
|1| 0.96424|
|2 |0.96485|
|3 |0.96449|
|4 |0.96512|
|5| 0.96393|

The running time for each split is less than 20 seconds on a RazorBlade 15, 2.3 GHz and 16 GB memory. The majority of the time taken to build this model was in defining the vocabulary in R (about1 minute for each Lasso Regression) and vectorizing the texts in Python using the scikit-learn library (about 5 minutes).
  
## 6. Conclusion
- From our experience with running the Neural Network model, it is very easy to obtain AUC’s that are larger than 0.9. For example, simply by taking the 1000 most common words in the reviews (after excluding common stop words), we could achieve a AUC of 0.93. However, increasing this performance was a little tricky and it mainly depends on the choice of the vocabulary. After trying a few methods, including the t-test on the vocabulary, we achieved a AUC of 0.955, but could not achieve the 0.96 benchmark. Then, we decided to try Prof. Liang's suggestion and run the vocabulary selection in R with Lasso. That made a substantial difference and we were able to achieve the benchmark.
  
- Regarding interpretability, it is specially hard to interpret the weights assigned to most neural network models. However, we did notice that decreasing the dimensionality N of the second (hidden layer) has a big impact on the performance of the model. For instance, the performance drops substantially for N < 4. We conjectured that the second layer is capturing different sentiments from correlation between words, and assigning weights to these intermediary sentiments. These intermediary sentiments may not be captured by a neural network with only two layers, which is equivalent to a Logistic Regression model. However, since many teams directly utilize Logistic Regression as their final model and it surpasses the benchmark, it is possible that with a better selection of the vocabulary, the second layer would not be necessary.
