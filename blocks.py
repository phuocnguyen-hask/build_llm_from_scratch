import torch
import torch.nn as nn
class MultiHeadAttention(nn.Module):
    def __init__(self, cfg, qkv_bias=False):
        super().__init__()
        d_in = cfg['n_embd']
        d_out = cfg['n_embd']
        self.n_head = cfg['n_head']
        
        assert d_out % self.n_head == 0

        self.head_dim = d_out // self.n_head
        self.W_keys = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_queries = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_values = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x, attn_mask=None):
        b, n_tokens, dim = x.shape
        # Step 1: Calculate keys, queries, values for batch
        keys = self.W_keys(x)
        queries = self.W_queries(x)
        values = self.W_values(x)

        # Step 2: calculate attn_scores, feed it to softmax, mask it, get weight_scores
        keys = keys.view(b, n_tokens, self.n_head, self.head_dim)
        queries = queries.view(b, n_tokens, self.n_head, self.head_dim)
        values = values.view(b, n_tokens, self.n_head, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(-1, -2)
        subsequent_mask = torch.tril(torch.ones(n_tokens, n_tokens), diagonal=0)
        attn_scores.masked_fill_(subsequent_mask == 0, -torch.inf) # mask subsequent tokens
        if attn_mask is not None: # mask padded tokens
            attn_scores.masked_fill_(~attn_mask, -torch.inf)
        weight_scores = torch.softmax(attn_scores, dim=-1)

        weighted_sum = weight_scores @ values

        weighted_sum = weighted_sum.reshape(b, n_tokens, -1)
        return weighted_sum
class FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_in = cfg['n_embd']
        d_out = cfg['n_embd']
        self.linear1 = nn.Linear(d_in, d_out)
        self.linear2 = nn.Linear(d_out, d_out)
    
    def forward(self, x):
        x = self.linear1(x)
        x = nn.GELU()(x)
        x = self.linear2(x)
        return x
class InputPreprocess(nn.Module):
    def __init__(self, tokenizer, cfg):
        super().__init__()
        self.tokenizer = tokenizer
        n_vocab = cfg['vocab_size']
        n_dim = cfg['n_embd']
        n_context = cfg['n_ctx']
        
        self.embedding_layer = nn.Embedding(n_vocab, n_dim)
        self.pos_embedding_layer = nn.Embedding(n_context, n_dim)
    
    def forward(self, example_list):
        ids = [self.tokenizer.encode(example) for example in example_list]
        max_length = max(len(id) for id in ids)
        attn_mask = [[1] * len(id) + [0] * (max_length - len(id)) for id in ids]
        attn_mask = torch.tensor(attn_mask, dtype=torch.bool)
        ids = [id + [0] * (max_length - len(id)) for id in ids]
        ids = torch.tensor(ids, dtype=torch.long) # change to torch tensor for compapility
        attn_mask = attn_mask.reshape(ids.size(0), 1, 1, max_length)

        embd_vecs = self.embedding_layer(ids)
        pos_embd_vecs = self.pos_embedding_layer(torch.arange(max_length)).unsqueeze(0)

        return embd_vecs + pos_embd_vecs, attn_mask

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_in = cfg['n_embd']
        d_out = cfg['n_embd']
        self.layer_norm1 = nn.LayerNorm(d_in)
        self.mhma = MultiHeadAttention(cfg)
        self.drop1 = nn.Dropout(cfg['embd_pdrop'])
        self.layer_norm2 = nn.LayerNorm(d_out)
        self.ffn = FFN(cfg)
        self.drop2 = nn.Dropout(cfg['embd_pdrop'])
    
    def forward(self, x, attn_mask=None):
        x_pre = x
        x_p1 = self.layer_norm1(x)
        x_p1 = self.mhma(x_p1, attn_mask)
        x_p1 = self.drop1(x_p1)
        x_p1 = x_pre + x_p1
        x_p2 = self.layer_norm2(x_p1)
        x_p2 = self.ffn(x_p2)
        x_p2 = self.drop2(x_p2)
        x_final = x_p1 + x_p2

        return x_final
