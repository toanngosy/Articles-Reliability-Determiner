import pandas as pd
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
stopwords = set(stopwords.words("english"))
porter = PorterStemmer()

def token(text):
    words = word_tokenize(text.decode('utf-8')) # split words
    words = [w.lower() for w in words if w.isalpha()] #get rif of punctuation
    words = [w for w in words if not w in stopwords]
    stemmed = [porter.stem(w) for w in words]
    w = " ".join(stemmed) 
    return w



