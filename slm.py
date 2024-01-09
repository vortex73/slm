 # !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('input.txt','r',encoding='utf-8') as file:
    text = file.read()

print(text[:1000])

l = sorted(list(set(text)))
size = len(l)
print("".join(l))
print(size)

stoi = {ch:i for i,ch in enumerate(l) }
print(stoi)

stoi = {ch:i for i,ch in enumerate(l)}
itos = {i:ch for i,ch in enumerate(l)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda n: ''.join([itos[c] for c in n])

encoded = encode("This is so cool")
print(encoded)
decode(encoded)



import torch
data = torch.tensor(encode(text),dtype=torch.long)
print(data.shape,data.dtype)
print(data[:1000])

num = int(0.9*len())
train = data[:num]
test = data[num:]

blksiz = 8
train[:blksiz+1]

len(data)
torch.manual_seed(1337)
parallels = 4
bloc = 8

def batchhus(split):
    temp = train if split=='train' else test
    offset = torch.randint(len(temp)-bloc,(parallels,))
    x = torch.stack([temp[i:i+bloc] for i in offset])
    y = torch.stack([temp[i+1:i+bloc+1] for i in offset])
    return x,y

xb,yb = batchhus('train')
print(xb)
print(yb)

for a in range(parallels):
    for b in range(bloc):
        context = xb[a,:b+1]
        result = yb[a,b]
        print(f"Context : {context.tolist()} -> result : {result}")

import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class bigram(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self,idx,targets):
        logits = self.token_embedding_table(idx)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits,targets)
        return logits,loss

m = bigram(size)
logits, loss = m(xb,yb)
print(logits.shape)
print(loss)

