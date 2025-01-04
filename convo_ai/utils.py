# from nltk.stem.porter import PorterStemmer

import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np


lemmatizer = WordNetLemmatizer()
def tokenize(sentence):
    """Tokenize a sentence into words."""
    return nltk.word_tokenize(sentence)

def bag_of_words(tokenized_sentence, all_words):
    """Create a bag-of-words representation."""
    tokenized_sentence = [lemmatizer.lemmatize(word.lower()) for word in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# stemmer = PorterStemmer()
# def stem(word):
#     return stemmer.stem(word.lower())

# def lemmatize(word):
#     return lemmatizer.lemmatize(word.lower())
