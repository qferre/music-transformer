import torch
from torch import nn
import math, time

from lib import data_acquisition as da
from lib import music_transformer

from torchtext.vocab import vocab
from collections import OrderedDict

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### --------------------------- DATA --------------------------------------- ###
train_iter = da.MusicYielder()
seq_len = 15                 # desired length of input sequences to be generated

# TODO : for now, tokens are manually specified, but replace it with full 
# Standard English when we process the full melody files.
tokens = ['A','B','C','D','E','F','G']
unk_token = '<unk>'
default_index = -1
myvocab = vocab(
    OrderedDict([(token, 1) for token in tokens]), specials=[unk_token]
    )
myvocab.set_default_index(default_index)
myvocab.set_default_index(myvocab[unk_token])


### --------------------------- MODEL DEFINITION --------------------------- ###
"""
This model takes a sequence of letters (musical notes) and tries to predict the next one.
"""
ntokens = len(myvocab)  # Size of vocabulary
emsize = 20             # Embedding dimension
d_hid = 15              # Dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2             # Number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2               # Number of heads in nn.MultiheadAttention
dropout = 0.2           # Dropout probability
model = music_transformer.TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)


### --------------------------- TRAINING THE MODEL ------------------------- ###
lr = 0.01               # Learning rate (initial)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5.0, gamma=0.999)

batch_size = 100             # How many (sequence, target) pairs per batch
nb_batches_training = 1500   # how many batches until training is complete


def train(model: nn.Module) -> None:
    model.train() # Turn on 'train' mode
    total_loss = 0.
    log_interval = 10
    start_time = time.time()   

    losses = []

    # For each batch...
    for i in range(nb_batches_training):

        ## Forward pass

        # Generate a data batch
        data, targets = train_iter.produce_batch(batch_size = batch_size,
            seq_len = seq_len, 
            vocab = myvocab, device = device)

        # This mask prevents information flow from the future and must be passed to the Transformer
        src_mask = music_transformer.generate_square_subsequent_mask(seq_len).to(device)

        # Predict next note for each seq of the current batch
        output = model(data, src_mask)

        # Compute loss
        loss = criterion(output.view(-1, ntokens), targets) 

        ## Backpropagation pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        ## Debug print
        total_loss += loss.item()
        losses += [loss.item()] # Remember the losses
        if i % log_interval == 0 and i > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| {i:5d}/{nb_batches_training:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()     

    return losses

## Run the training
losses = train(model)
plt.plot(losses) ; plt.show()




### --------------------------- SAVE MODEL --------------------------------- ###

model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('./models/model_scripted.pt') # Save


