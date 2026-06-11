
import torch
from torch import nn 
import tiktoken
import matplotlib.pyplot as plt
import torch.nn.functional as F 

if torch.cuda.is_available:
    device='cuda'
else:
    device='cpu'


with open('Week3_work/input.txt','r',encoding='utf-8') as f:
    text=f.read()

# We will use tiktoken with the GPT-2 encoding for a smaller, more manageable vocabulary.
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)

# print(f"Using tiktoken with vocab size: {vocab_size} on device: {device}")

# Encode the entire text dataset at once into a PyTorch tensor
data = torch.tensor(encode(text), dtype=torch.long)
# print(f"Total tokens in dataset: {len(data)}")

# Using tiktoken with vocab size: 50257 on device: cuda
# Total tokens in dataset: 338024 



# Split data sequentially to preserve text order: 90% train, 10% validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

#hyper_variables
batch_size=64 # how many independent sequences will we process in parallel?
block_size=256 # what is the maximum context length for predictions?
n_layer=6 # no of transformers in the model
n_embd=384 # no of numbers in the embedding vector 
n_heads=6 # no of active calculation heads/self attention blocks in the single transformer
n_kv_heads=2 # no of key and val pair for the single transformer because of the group query attention
learning_rate=3e-4 # rate at which the model will learn
dropout=0.1 # dropout rate drops neuron 10% of the time
eval_iters = 200 # no of evaluation iterations
max_iters=5000
eval_interval=500

class RotaryEmbedding(nn.Module):
    def __init__(self,head_size, max_seq_len=2048, theta=10000.0):
        super().__init__()

        self.head_size = head_size
        
        # Generate inverse frequencies for rotation angles: shape (head_size // 2)
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_size, 2).float() / head_size))
        
        # Generate position index axis steps: shape (max_seq_len)
        t = torch.arange(max_seq_len, dtype=torch.float)
        
        # Matrix outer-product multiplication to form angle frequencies grid
        # freqs shape: (max_seq_len, head_size // 2)
        freqs = torch.outer(t, inv_freq)
        
        # Duplicate each column to map matching pairs across full head size
        # emb shape: (max_seq_len, head_size)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache cosine and sine variations as buffers
        self.register_buffer("cos_cached", emb.cos()) # (max_seq_len, head_size)
        self.register_buffer("sin_cached", emb.sin()) # (max_seq_len, head_size)

    def _rotate_half(self, x):
        # Splits the head vector elements in half and rotates their sign
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, seq_len):
        # x shape: (B, n_head, T, head_size)
        # Pull matching slices from cached rotation tensors matching token length T
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(1) # (1, 1, T, head_size)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(1) # (1, 1, T, head_size)
        
        # Apply the explicit rotation transformation matrix operation shortcut
        return (x * cos) + (self._rotate_half(x) * sin)

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class GroupQueryAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.head_size = head_size
        self.num_queries_per_kv = n_heads // n_kv_heads

        # Projection dimensions map according to head groups
        # NOTE: The output dimension must be the total size for all heads combined.
        self.query = nn.Linear(n_embd, n_heads * head_size, bias=False)
        self.key = nn.Linear(n_embd, n_kv_heads * head_size, bias=False)
        self.value = nn.Linear(n_embd, n_kv_heads * head_size, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

        # Causal mask tracking matrix
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # Rotary embedding block integrated into attention
        self.rope = RotaryEmbedding(head_size)

        # add the dropout for better realibile traning of the model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # Project and reshape Q, K, V for grouped-query attention
        q = self.query(x).view(B, T, n_heads, self.head_size).transpose(1, 2)      # (B, n_heads, T, head_size)
        k = self.key(x).view(B, T, n_kv_heads, self.head_size).transpose(1, 2)    # (B, n_kv_heads, T, head_size)
        v = self.value(x).view(B, T, n_kv_heads, self.head_size).transpose(1, 2)  # (B, n_kv_heads, T, head_size)

        # Apply RoPE angular changes to queries and keys
        q = self.rope(q, T)
        k = self.rope(k, T)

        # Repeat K and V to match the number of query heads for GQA
        k = torch.repeat_interleave(k, repeats=self.num_queries_per_kv, dim=1) # (B, n_heads, T, head_size)
        v = torch.repeat_interleave(v, repeats=self.num_queries_per_kv, dim=1) # (B, n_heads, T, head_size)

        # Compute attention scores ("affinities")
        # (B, n_heads, T, head_size) @ (B, n_heads, head_size, T) -> (B, n_heads, T, T)
        score = q @ k.transpose(-2, -1) * (self.head_size**-0.5)
        score = score.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, n_heads, T, T)
        weights = torch.softmax(score, dim=-1) # (B, n_heads, T, T)
        weights = self.dropout(weights)

        # Perform the weighted aggregation of the values
        # (B, n_heads, T, T) @ (B, n_heads, T, head_size) -> (B, n_heads, T, head_size)
        out = weights @ v

        # Reconstruct output to original embedding shape
        # (B, n_heads, T, head_size) -> (B, T, n_heads, head_size) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)
    

class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads=nn.ModuleList(GroupQueryAttentionHead(head_size) for _ in range(num_heads))
        self.proj=nn.Linear(head_size*num_heads,n_embd)
        self.dropout=nn.Dropout(dropout)

        def forward(self,x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
            return out
        

class SwiGLUFeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # Calculate hidden dimension size according to the 8/3 scaling rule
        hidden_dim = int(2 * (4 * n_embd) / 3)
        
        self.w_gate = nn.Linear(n_embd, hidden_dim, bias=False)  # Gate pathway
        self.w_up   = nn.Linear(n_embd, hidden_dim, bias=False)  # Value pathway
        self.w_down = nn.Linear(hidden_dim, n_embd, bias=False)  # Down projection

    def forward(self, x):
        # x shape: (B, T, n_embd)
        # F.silu is the SiLU/Swish activation function
        gate_output = F.silu(self.w_gate(x)) 
        up_output = self.w_up(x)
        
        # Element-wise multiply the parallel pathways
        combined = gate_output * up_output # Shape: (B, T, hidden_dim)
        
        return self.w_down(combined) # Shape: (B, T, n_embd)

# class FeedForward(nn.Module):
#     """ a simple linear layer followed by a non-linearity """

#     def __init__(self, n_embd):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(n_embd, 4 * n_embd),
#             nn.ReLU(),
#             nn.Linear(4 * n_embd, n_embd),
#             nn.Dropout(dropout),
#         )

#     def forward(self, x):
#         return self.net(x)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # Gain parameter (gamma) initialized to 1s
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x shape: (B, T, C)
        # Compute variance (mean of squared values) along the last dimension
        variance = x.pow(2).mean(-1, keepdim=True)
        
        # Normalize and scale by learnable weight parameter
        # torch.rsqrt(x) computes 1 / sqrt(x) directly for speed
        return x * torch.rsqrt(variance + self.eps) * self.weight

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_heads):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = GroupQueryAttentionHead(head_size)
        self.ffwd = SwiGLUFeedForward(n_embd)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class MY_Gpt(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,n_embd)
        self.blocks=nn.Sequential(*[Block(n_embd, n_heads=n_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
       if isinstance(module, nn.Linear):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
          if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
       elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.embedding(idx) # (B,T,C)
        x = self.blocks(tok_emb) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

    def save(self, filepath='my_gpt_model.pth'):
        """Saves the model's state dictionary to a file."""
        torch.save(self.state_dict(), filepath)
        print(f"Model state dictionary saved to {filepath}")

model = MY_Gpt()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# Save the trained model
m.save()