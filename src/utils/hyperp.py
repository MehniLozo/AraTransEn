from train import BATCH_SIZE

class Hyperparam:
    BATCH_SIZE = 16
    learning_rate = 0.0001
    num_epochs = 30

    nheads = 8
    nencoder_layers = 3
    ndecoder_layers = 3

    len_max = 230
    dropout = 0.4
    emb_size = 256