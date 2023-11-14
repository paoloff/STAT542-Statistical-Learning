import pandas as pd
import numpy as np

######## Step I: train the model ########

# Read data into Pandas table:
data = pd.read_csv("Train.csv")

# Define dictionaries where probabilities will be stored.
# First, for Churn = 0:
dict0 = {}

# Then, for Churn = 1:
dict1 = {}

# Get the names of the categorical prediction variables:
keys = data.keys()[0:-1]

# Initialize dictionaries with 0s for all entries:
for key in keys:
	dict0[key] = {}
	dict1[key] = {}
	values = set(data[key])
	for element in values:
		dict0[key][element]=(0)
		dict1[key][element]=(0)

# Obtain the count number for each value of each predictor variable.
# For dict0, restrict to the cases where Churn = 0
# For dict1, restrict to the cases where Chrun = 1
# At the same time, count the number of incidences for Churn = 0

c = 0

for i in range(len(data)):
	if data.iloc[i]["Churn"] == 0:
		c += 1
		for key in keys:
			dict0[key][data.iloc[i][key]] += 1
	else:
		for key in keys:
			dict1[key][data.iloc[i][key]] += 1

# Normalize the counts to obtain the conditional probabilities:
for key in keys:
	sum_ = sum(dict0[key].values())
	for element in dict0[key]:
		dict0[key][element] = dict0[key][element]/sum_

	sum_ = sum(dict1[key].values())
	for element in dict1[key]:
		dict1[key][element] = dict1[key][element]/sum_

# Finally, evaluate probabilities of Churn = 0 and Churn = 1
prob0 = c/(c+len(data))
prob1 = 1 - prob0


######## Step II: evaluate the model on the train set ########

predicted_train = []

# Predict response row by row:
for i in range(len(data)):

	P0 = np.log(prob0)
	P1 = np.log(prob1)

	for key in keys:
		P0 += np.log(dict0[key][data.iloc[i][key]])
		P1 += np.log(dict1[key][data.iloc[i][key]])

	predicted_train.append(np.argmax([P0,P1]))

# Print accuracy:
print("Accuracy for the train data: ", sum(data["Churn"]==predicted_train)/len(data), "\n")


######## Step III: evaluate the model on the validation set ########

# Read the data matrix:
validation = pd.read_csv("Validation.csv")
predicted_val = []

# Predict response row by row:
for i in range(len(validation)):

	P0 = np.log(prob0)
	P1 = np.log(prob1)

	for key in keys:
		P0 += np.log(dict0[key][validation.iloc[i][key]])
		P1 += np.log(dict1[key][validation.iloc[i][key]])

	predicted_val.append(np.argmax([P0,P1]))


# Compute confusion matrix
tp = 0; tn = 0; fp = 0; fn = 0;

for i in range(len(validation)):
	if validation["Churn"][i] == 1 and predicted_val[i] == 1:
		tp += 1
	elif validation["Churn"][i] == 1 and predicted_val[i] == 0:
		fn += 1
	elif validation["Churn"][i] == 0 and predicted_val[i] == 1:
		fp +=1
	else:
		tn += 1

# Print metrics:
print("Accuracy for the validation data: ", sum(validation["Churn"]==predicted_val)/len(validation), "\n")
print("Confusion Matrix for validation set:","\n" )
print("True Positive = ", tp, "; False Negative = ", fn, "\n")
print("False Positive = ", fp, "; True Negative = ", tn, "\n")
print("Precision for validation set = ", tp/(tp+fp), "\n")
print("Recall for validation set = ", tp/(tp+fn), "\n")
print("F1 score for validation set ", 2*((tp/(tp+fp))*tp/(tp+fn))/((tp/(tp+fp)+tp/(tp+fn))))

