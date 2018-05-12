from nltk import ngrams
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

def train_LR_classifier(X, y):
    classifier = LogisticRegression(C=10)
    classifier.fit(X, y) 
    return classifier

def evaluate(X, yt, classifier):
    yp = classifier.predict(X)
    accuracy = metrics.accuracy_score(yt, yp)
    print("  Accuracy", accuracy)

def split_validation_set(X, y, valid_size):
	X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_size, random_state=1)
	return X_train, X_valid, y_train, y_valid

def tfidf_ngrams(X, X_valid, n):
	vectWord = TfidfVectorizer(lowercase=True, analyzer='word',ngram_range=(1,n),dtype=np.float32)
	vectorX = vectWord.fit_transform(X)
	vectorValidX = vectWord.transform(X_valid)
	return vectorX, vectorValidX
