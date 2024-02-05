import torch
import torch.nn as nn 
from config import device 

class Transformer(nn.Module):
    def __init__(
        self,
        emb_size,
        svocab_size, # source vocab
        tvocab_size, ## target vocabulary
        spad_idx,
        nheads,
        nencoder_layers, # number of encoder layers
        ndecoder_layers,
        forward_exp,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.sembeddings = nn.Embedding(svocab_size,emb_size)
        self.spositional_embeddings= nn.Embedding(max_len,emb_size)
        self.tembeddings= nn.Embedding(tvocab_size,emb_size)
        self.tpositional_embeddings= nn.Embedding(max_len,emb_size)
        self.device = device
        self.transformer = nn.Transformer(
            emb_size,
            nheads,
            nencoder_layers,
            ndecoder_layers,
        )
        self.fc_out = nn.Linear(emb_size, tvocab_size)
        self.dropout = nn.Dropout(dropout)
        self.spad_idx = spad_idx
    def smasking(self,src):
        ## making a source mask
        smask= src.transpose(0,1) == self.spad_idx
        return smask
    
    def forward(self,src,trg):
        sseq_len, S = src.shape
        tseq_len, S = trg.shape
        spositions = (
            torch.arange(0, sseq_len).unsqueeze(1).expand(sseq_len, S).to(self.device)
        )
        tpositions = (
            torch.arange(0, tseq_len).unsqueeze(1).expand(tseq_len, S).to(self.device)
        )
        # source embedding 
        sembed = self.dropout(
            ( self.sembeddings(src) + self.spositional_embeddings(spositions) )
        )
        # target embedding 
        tembed = self.dropout(
                ( self.tembeddings(trg) + self.trg_positional_embeddings(tpositions) )
        )
        # source masking
        spadding_mask = self.smasking(src)
        # target masking
        tmask = self.transformer.generate_square_subsequent_mask(tseq_len).to(device)

        out = self.transformer(sembed,tembed,src_key_padding = spadding_mask, tgt_mask = tmask )
        out = self.fc_out(out)
        return out