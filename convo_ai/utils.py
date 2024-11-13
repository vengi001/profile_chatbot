import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def tokenize(sentence):
    tokenized_sentence = nltk.word_tokenize(sentence)
    return tokenized_sentence


def bag_of_words(tokenized_sentence, all_words):
    tokenized_stemmer = [lemmatizer.lemmatize(word.lower()) for word in tokenized_sentence]
    bag_of_words = np.zeros(len(all_words), dtype=np.float32)
    
    for index, word in enumerate(all_words):
        if word in tokenized_stemmer:
            bag_of_words[index] = 1.0
    
    return bag_of_words

def stem(word):
    return stemmer.stem(word.lower())

def lemmatize(word):
    return lemmatizer.lemmatize(word.lower())
