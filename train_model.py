import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device if device == "gpu" else "Warning : currently running on the cpu!")


block_size = 64 # 64
batch_size = 128 # 128

n_embd = 384
n_layer = 8
n_head = 8

max_iters = 4000
eval_iters = 100 # (iters / batch_size) is a good value

learning_rate = 3e-4

dropout = 0.2 #20%

chars = ""
with open("dataset/vocab.txt", 'r', encoding="utf-8") as f:
    text = f.read()
    chars = sorted(set(text))

vocab_size = len(chars)

## encoder and decoder
string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

## get batch
# memory map used to get random chunk without opening HUGE file
def get_random_chunk(split):
    filename = "dataset/train_split.txt" if split == "train" else "dataset/val_split.txt"
    with open(filename, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, file_size - block_size * batch_size)

            mm.seek(start_pos)
            block = mm.read(block_size * block_size - 1)

            decoded_block = block.decode("utf-8", errors="ignore").replace("\r",
                                                                           "")  # ignore errors incase corruption , replace got errors about \r
            data = torch.tensor(encode(decoded_block), dtype=torch.long)

    return data


def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

## estimate loss

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() #turn on eval mode
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() #turn on train mode
    return out


### ----GPT MODEL---- ###

class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        # input of size (B, T, C)
        # output of size (B, T, head_size)
        B, T, C = X.shape
        K = self.key(X)  # (B, T, head_size)
        Q = self.query(X)  # (B, T, head_size)
        # compute attention scores ("affinities")
        weights = Q @ K.transpose(-2, -1) * K.shape[
            -1] ** -0.5  # (B, T, head_size) @ (B, head_size, T) => (B, T, T) transpose swaps dims
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)  # (B, T, T)
        weights = self.dropout(weights)
        # perform the weighted aggregation of the values
        V = self.value(X)
        out = weights @ V  # (B, T, T) @ (B, T, head_size) => (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads,
                              n_embd)  # projection: project head_size * num_heads to n_embd to prevent errors in the future
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        out = torch.cat([h(X) for h in self.heads],
                        dim=-1)  # cat along last dim (B, T, C) cat along channel dim (B , T, [h1 h1 h2 h2 h3 h3 h4 h4])
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """a simple linear layer followed by a non linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # switching the n_embd and 4 * n_embd makes output be n_embd by n_embd
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # switching the n_embd and 4 * n_embd makes output be n_embd by n_embd
            nn.Dropout(dropout),  # set some neurons to 0 to prevent overfitting
        )

    def forward(self, X):
        return self.net(X)


class Block(nn.Module):
    """ Transformer Block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: number of heads in parralel
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, X):
        Y = self.sa(X)
        X = self.ln1(X + Y)
        Y = self.ffwd(X)
        X = self.ln2(X + Y)

        return X


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emmbeding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])  # decoder chain

        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_emmbeding_table(index)  # (B, T, C)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device))  # (T, C)
        X = tok_emb + pos_emb  # (B, T, C)
        X = self.blocks(X)  # (B, T, C)
        X = self.ln_f(X)  # (B, T, C)
        logits = self.lm_head(X)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indecies in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self.forward(index)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)            negative index => loop around to last index
            # apply softmax
            probs = F.softmax(logits, dim=-1)  # (B, C)           negative index => loop around to last index
            # sample from the prob distribution
            index_next = torch.multinomial(probs, num_samples=1)
            # apppend sampled index to the sequence
            index = torch.cat((index, index_next), dim=1)  # (B, T+1) => becomes larger when time gets larger => becomes larger with every char
        return index


model = GPTLanguageModel(vocab_size)

#print("loading model...")
#with open("model-0.pkl", "rb") as f:
#    model = pickle.load(f)
m = model.to(device)
#print("loaded")

### train loop ###
# create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step {iter}, train loss: {losses["train"]:.3f} val loss: {losses["val"]:.3f}")

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model.forward(xb, yb)  # xb = index yb = targets
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

print("saving model...")
with open("model-0.pkl", "wb") as f:
    pickle.dump(model, f)
print("model saved")