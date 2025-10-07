import torch
from torch.utils.data import Dataset
class JonahDataset(Dataset):
    def __init__(self, txt, tokenizer, context_length, stride):
        
        self.tokens = tokenizer.encode(txt)

        self.inputs = []
        self.targets = []

        tokens_length = len(self.tokens)
        for i in range(0, tokens_length - context_length, stride):
            self.inputs.append(torch.tensor(self.tokens[i: i + context_length]))
            self.targets.append(torch.tensor(self.tokens[i+1:i+context_length +1]))
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
