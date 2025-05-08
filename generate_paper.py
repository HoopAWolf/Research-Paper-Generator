from model import ResearchLLM
from tokenizers import Tokenizer
import pandas as pd

def generate_research_paper(model, tokenizer, data_path=None, title="Study", abstract="N/A", max_length=2000):
    model.eval()
    data = pd.read_csv(data_path) if data_path else None
    data_desc = data.to_string() if data is not None and not data.empty else "No tabular data provided."
    prompt = (
        f"[SEP]Title: {title}\n"
        f"[SEP]Abstract: {abstract}\n"
        f"[SEP]Data: {data_desc}\n"
        f"[SEP]Introduction:"
    )
    tokens = tokenizer.encode(prompt).ids
    print(f"Prompt length: {len(tokens)}")  # Log length
    paper = model.generate(tokenizer, prompt, max_length=max_length)
    return paper