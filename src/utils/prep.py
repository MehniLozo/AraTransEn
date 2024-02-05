import string
import pandas as pd
import regex as re
import nltk
from unicodedata import normalize
from pickle import load
from pickle import dump
from collections import Counter

def load(name):
    file = open(name, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text

def senticize(doc):
	return doc.strip().split('\n')

def len_sents(sentences):
	l = [len(s.split()) for s in sentences]
	return min(l), max(l)

    