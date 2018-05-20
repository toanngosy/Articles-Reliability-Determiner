from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np 

def evaluate(X,y, classifer):
    pred = classifer.predict(X)
    accuracy  = metrics.accuracy_score(y, pred)
    print("Accuracy: ", accuracy)

def train_xgboost(X,y):
    classifier = XGBClassifier()
    classifier.fit(X,y)
    return classifier

def split_validation_set(X, y, valid_size):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_size, random_state=1)
    return X_train, X_valid, y_train, y_valid

def tfidf_ngrams(X, X_valid, n):
    vectWord = TfidfVectorizer(lowercase=True, analyzer='word',ngram_range=(1,n),dtype=np.float32)
    vectorX = vectWord.fit_transform(X)
    vectorValidX = vectWord.transform(X_valid)
    return vectorX, vectorValidX

def run_xgboost(X,y, valid_size= 0.3, n_gram = 2):
    print("Run Xgboost model...")
    X_train, X_valid, y_train, y_valid = split_validation_set(X,y, valid_size)
    X_train_vector, X_valid_vector = tfidf_ngrams(X_train, X_valid, n_gram)
    classifier = train_xgboost(X_train_vector, y_train)
    print("Evaluate on the training set")
    evaluate(X_train_vector, y_train, classifier)
    print("Evaluate on the valid set")
    evaluate(X_valid_vector, y_valid,classifier)

