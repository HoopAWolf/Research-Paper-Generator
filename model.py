import torch
import torch.nn as nn
import math

class ResearchLLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        pe = self.positional_encoding(src.size(1), self.d_model).to(src.device)
        src = src + pe
        output = self.transformer(src, src_mask)
        return self.fc(output)

    def generate(self, tokenizer, prompt, max_length=500, temperature=0.7):
        self.eval()
        tokens = tokenizer.encode(prompt).ids
        input_ids = torch.tensor([tokens], dtype=torch.long).to(next(self.parameters()).device)
        generated = input_ids

        for _ in range(max_length - len(tokens)):
            with torch.no_grad():
                logits = self(generated)[:, -1, :] / temperature
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == tokenizer.encode("[EOS]").ids[0]:
                    break
        return tokenizer.decode(generated[0].tolist())