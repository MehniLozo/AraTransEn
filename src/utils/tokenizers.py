import re
import pandas as pd
import random
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.ar import Arabic

from torchtext import data
from torchtext.legacy import data


seeding = 32
spacy_eng = spacy.load("en_core_web_sm")
arab = Arabic()
ar_tk = Tokenizer(arab.vocab)

def tk_en(content):
  return [w.text for w in spacy_eng.tokenizer(content)]

def tk_ar(content):
  return [w.text for w in
  ar_tk(re.sub(r"\s+"," ",re.sub(r"[\.\'\"\n+]"," ",content)).strip())]


SRC = data.Field(tokenize=tk_en,batch_first=False,init_token="<sos>",eos_token="<eos>")
TRG = data.Field(tokenize=tk_ar,batch_first=False,tokenizer_language="ar",init_token="بداية",eos_token="نهاية")

class Textsizing(data.Dataset):
  def __init__(self, df, src_field, target_field, is_test=False, **kwargs):
    fields = [('en', src_field), ('ar',target_field)]
    samples = []
    for i, r in df.iterrows():
      en = r.en
      ar = r.ar
      samples.append(data.Example.fromlist([en, ar], fields))
      super().__init__(samples, fields, **kwargs)
    
    def __len__(self):
      return len(self.samples)
    def __getitem__(self,idx):
      return self.samples[idx]
    


if __name__ == "__main__":
    df = pd.read_csv("data/arabic_english.txt",delimiter="\t",names=["en","ar"])
    torchdataset = Textsizing(df,SRC,TRG)

    train_data, valid_data = torchdataset.split(split_ratio=0.8, random_state = random.seed(32))

    SRC.build_vocab(train_data,min_freq=2)
    TRG.build_vocab(train_data,min_freq=2)

    print(train_data[1].__dict__)