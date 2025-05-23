import torch
from torch.amp import autocast, GradScaler
from model import ResearchLLM
from data_prep import prepare_data, create_dataloader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    tokenizer, wikitext, papers, vocab_size = prepare_data()
    print(f"Vocab size: {vocab_size}")
    model = ResearchLLM(vocab_size=vocab_size, max_length=2048).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))
    scaler = GradScaler() if device.type == "cuda" else None

    pretrain_loader = create_dataloader(wikitext, tokenizer, batch_size=4, max_length=2048)
    model.train()
    for epoch in range(5):
        for batch in pretrain_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda', enabled=device.type == "cuda"):
                output = model(batch[:, :-1])
                loss = criterion(output.transpose(1, 2), batch[:, 1:])
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        logger.info(f"Pretrain Epoch {epoch+1}, Loss: {loss.item()}")

    finetune_loader = create_dataloader(papers, tokenizer, batch_size=2, max_length=2048)
    for epoch in range(5):
        for batch in finetune_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda', enabled=device.type == "cuda"):
                output = model(batch[:, :-1])
                loss = criterion(output.transpose(1, 2), batch[:, 1:])
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        logger.info(f"Finetune Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "research_llm.pt")
    return model, tokenizer

if __name__ == "__main__":
    train_model()