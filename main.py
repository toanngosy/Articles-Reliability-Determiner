import pandas as pd
from preprocess import token
from LR import train_LR_classifier, evaluate, split_validation_set, tfidf_ngrams
#TA ngu vl 

#import dataset
train = pd.read_csv("data/train.csv",encoding = "ISO-8859-1")
test = pd.read_csv("data/test.csv",encoding = "ISO-8859-1")

#import the processed text
train_processed= pd.read_csv("data/train_processed.csv", encoding = "ISO-8859-1")
test_processed= pd.read_csv("data/test_processed.csv",encoding = "ISO-8859-1")

#concatenate the processed files and the initial dataset
train = pd.concat([train, train_processed], axis = 1,join_axes=[train.index])
train = train.drop("Unnamed: 0", axis = 1)
test = pd.concat([test, test_processed], axis = 1,join_axes=[test.index])
test = test.drop("Unnamed: 0", axis = 1)
train.fillna(' ',inplace=True)
test.fillna(' ',inplace=True)

###Logistic regression with only text###
X_train, X_valid, y_train, y_valid = split_validation_set(train.Processed, train.label, 0.4)
X_train_vector, X_valid_vector = tfidf_ngrams(X_train, X_valid, 2)
LRclassifier = train_LR_classifier(X_train_vector, y_train)
evaluate(X_valid_vector, y_valid, LRclassifier)
