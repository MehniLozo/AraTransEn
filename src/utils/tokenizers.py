import re
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.ar import Arabic

seeding = 32
spacy_eng = spacy.load("en_core_web_sm")
arab = Arabic()
ar_tk = Tokenizer(arab.vocab)

def tk_en(content):
  return [w.text for w in spacy_eng.tokenizer(content)]

def tk_ar(content):
  return [w.text for w in
  ar_tk(re.sub(r"\s+"," ",re.sub(r"[\.\'\"\n+]"," ",content)).strip())]
