import pandas as pd


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

