import torch
import torch.nn as nn

class ResearchLLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, max_length=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.max_length = max_length

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        src = src + self.pos_encoder[:, :src.size(1), :]
        output = self.transformer(src, src_mask)
        return self.fc(output)

    def generate(self, tokenizer, prompt, max_length=500, temperature=0.7):
        self.eval()
        tokens = tokenizer.encode(prompt).ids
        print(f"Prompt tokens: {tokenizer.encode(prompt).tokens[:50]}")
        print(f"Prompt length: {len(tokens)}")
        eos_token_id = tokenizer.encode("[EOS]").ids[0]
        print(f"[EOS] token ID: {eos_token_id}")
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            print(f"Warning: Prompt truncated to {self.max_length} tokens")
        input_ids = torch.tensor([tokens], dtype=torch.long).to(next(self.parameters()).device)
        generated = input_ids
        initial_prompt_length = len(tokens)

        max_generated_tokens = min(max_length - initial_prompt_length, self.max_length - initial_prompt_length)
        if max_generated_tokens < 0:
            print(f"Warning: Prompt length ({initial_prompt_length}) exceeds max_length ({max_length})")
            return tokenizer.decode(tokens)

        for i in range(max_generated_tokens):
            with torch.no_grad():
                if generated.size(1) > self.max_length:
                    start_idx = max(0, generated.size(1) - self.max_length)
                    generated = generated[:, start_idx:]
                    print(f"Warning: Generated sequence truncated to {self.max_length} tokens")
                logits = self(generated)[:, -1, :] / temperature
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == eos_token_id:
                    print("EOS token detected, stopping generation")
                    break
        return tokenizer.decode(generated[0].tolist())