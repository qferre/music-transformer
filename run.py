import torch

from lib import data_acquisition as da
from lib import music_transformer

import seaborn as sns
import sklearn.metrics

from torchtext.vocab import vocab
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



### --------------------------- PARAMETERS --------------------------------- ###
# TODO: mutualize this with train.py, by putting them in a YAML file.
seq_len = 15

tokens = ['A','B','C','D','E','F','G']
unk_token = '<unk>'
default_index = -1
myvocab = vocab(
    OrderedDict([(token, 1) for token in tokens]), specials=[unk_token]
    )
myvocab.set_default_index(default_index)
myvocab.set_default_index(myvocab[unk_token])



### --------------------------- RELOAD MODEL ------------------------------- ###
model = torch.jit.load('./models/model_scripted.pt')
model.eval() # Turn on 'evaluation' mode

### --------------------------- EVALUATION --------------------------------- ###
evaluations = 2500

# TODO: separate training and testing data, obviously!
test_iter = da.MusicYielder()

data, targets = test_iter.produce_batch(evaluations, 
    seq_len = seq_len, vocab = myvocab, device = device)
src_mask = music_transformer.generate_square_subsequent_mask(seq_len).to(device)
output = model(data, src_mask)
prediction = torch.argmax(output, dim=1)

t = targets.detach().numpy()
p = prediction.detach().numpy()

cf_matrix = sklearn.metrics.confusion_matrix(t, p,
    normalize = 'true')
sns.heatmap(cf_matrix, annot=True)



### --------------------------- PRACTICE ----------------------------------- ###

"""
Make a prompt and try to see what it predicts.
"""

prompt = "A B B A C D E A A F A C C A E B B D E".split(" ")
desired_len = 100
result = prompt

while len(result) < desired_len:

    # Get the k last tokens
    raw_text = result[-seq_len:]

    Xraw = [torch.tensor(myvocab([item]), dtype=torch.long) for item in raw_text]
    X = [torch.cat(tuple(filter(lambda t: t.numel() > 0, Xraw)))]
    Xtensor = torch.stack(X, axis = 1) 

    output = model(Xtensor, src_mask)
    prediction = torch.argmax(output, dim=1)

    # Extract the only predicted value
    p = prediction.detach().numpy()
    p = p.tolist()[0]

    # Append prediction to result
    new_token = myvocab.lookup_token(p)    
    result += new_token

print("Generated string:", result)