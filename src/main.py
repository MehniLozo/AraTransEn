import numpy as np
import torch
from torch import optim, nn
from torchtext.legacy import data
from utils.tokenizers import SRC, TRG
from models.transformer_trans import Transformer
from utils.config import device
from utils.prep import train_data, valid_data
from utils.hyperp import BATCH_SIZE,num_epochs,  emb_size, len_max, dropout,spad_idx, learning_rate, nums_epochs, nheads, nencoder_layers, ndecoder_layers, forward_expansion

train_iter, valid_iter = data.BucketIterator.splits(
    (train_data, valid_data), 
    batch_size = BATCH_SIZE,
    sort = None,
    sort_within_batch = False,
    sort_key = lambda x: len(x.en),
    device = device,
    shuffle = True
)

svocab_size = len(SRC.vocab)
print("Size of EN vocab :", svocab_size)

tvocab_size = len(TRG.vocab)
print("Size of ES vocab :", tvocab_size)

model = Transformer(
    emb_size,
    svocab_size,
    tvocab_size,
    spad_idx,
    nheads,
    nencoder_layers, 
    ndecoder_layers, 
    forward_expansion,
    dropout,
    len_max,
    device
).to(device)

loss_check = []
loss_valid_check = []

optz= optim.Adam(model.parameters(), lr = learning_rate)
pad_idx = SRC.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

for epoch in range(num_epochs):
    stepLoss = []
    model.train()
    for batch in train_iter:
        input_data = batch.en.to(device)
        target = batch.ar.to(device)
        output = model(input_data, target[:-1])
        optz.zero_grad()

        output = output.reshape(-1,tvocab_size)
        target = target[1:].reshape(-1)
        loss = criterion(output, target)
        loss = criterion(output, target)
        loss.backward()
        optz.step()
        stepLoss.append(loss.item())

    loss_check.append(np.mean(stepLoss))
    print(" Epoch {} | Train Cross Entropy Loss: ".format(epoch), np.mean(stepLoss))
    with torch.no_grad():
        stepValidLoss = []
        model.eval()
        for i, batch in enumerate(valid_iter):
            inputs = batch.en.to(device)
            target = batch.ar.to(device)
            optz.zero_grad()
            output = model(inputs, target[:-1])
            output = output.reshape(-1, tvocab_size)
            target = target[1:].reshape(-1)
            loss = criterion(output,target)
            stepValidLoss.append(loss.item())
    loss_valid_check.append(np.mean(stepValidLoss))
    print("Epoch {} | Validation cross entrop loss : ".format(epoch),np.mean(stepValidLoss))
