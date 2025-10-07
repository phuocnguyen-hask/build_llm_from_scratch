import torch
import torch.nn as nn
from .blocks import TransformerBlock
class MyLLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding_layer = nn.Embedding(cfg['vocab_size'], cfg['n_embd'])
        context_length = cfg['task_specific_params']['text-generation']['max_length']
        self.pos_embedding_layer = nn.Embedding(context_length, cfg['n_embd'])
        self.transformers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg['n_layer'])])
        self.last_embedding_layer = nn.Linear(cfg['n_embd'], cfg['vocab_size'])
    
    def forward(self, x, attn_mask=None):
        size = x.shape[-1]
        x = self.embedding_layer(x)
        x += self.pos_embedding_layer(torch.arange(size))
        t_outputs = x
        for transformer in self.transformers:
            if attn_mask is not None:
                t_outputs = transformer(t_outputs, attn_mask)
            else:
                t_outputs = transformer(t_outputs)
        logits = self.last_embedding_layer(t_outputs)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        return probs # return probs to get full distribution not just generated token
