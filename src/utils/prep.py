import string
import pandas as pd
import regex as re
import nltk
arabic_stopwords = set(nltk.corpus.stopwords.words("arabic"))
arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''

punctuations = arabic_punctuations + string.punctuation


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

# stop words removal
def rem_swords(ct):
    filered = [w for w in ct.split() if w not in punctuations]
    return ' '.join(filered)

def clean(line):
    if (isinstance(line, float)):
        return None 
    line.replace('\n', ' ')
    line = ' '.join(line)
    trans = str.maketrans('','', punctuations)
    line = line.translate(trans)
    line = ' '.join(line)
    return line