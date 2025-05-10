from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import pandas as pd
import torch
import json
import os
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data(data_dir="data"):
    # Load WikiText
    wikitext = load_dataset("wikitext", "wikitext-103-v1", split="train[:5000]")
    wikitext_texts = [text for text in wikitext["text"] if isinstance(text, str) and text.strip()]
    logger.info(f"Loaded {len(wikitext_texts)} valid WikiText texts")

    # Load ArXiv papers
    def load_arxiv_file(file_path):
        papers = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    paper = json.loads(line.strip())
                    paper.setdefault("title", "Untitled")
                    paper.setdefault("abstract", "")
                    paper.setdefault("section_names", ["Introduction", "Methodology", "Results", "Discussion"])
                    paper.setdefault("sections", ["N/A"] * 4)
                    if len(paper["section_names"]) < 4:
                        paper["section_names"] = paper["section_names"] + ["N/A"] * (4 - len(paper["section_names"]))
                    if isinstance(paper["sections"], list):
                        paper["sections"] = [str(item) if not isinstance(item, list) else " ".join(str(subitem) for subitem in item) for item in paper["sections"]]
                    else:
                        paper["sections"] = ["N/A"] * 4
                    papers.append(paper)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in {file_path}: {line[:50]}... Error: {e}")
        return papers

    train_papers = load_arxiv_file(os.path.join(data_dir, "train.txt"))[:1000]
    if not train_papers:
        train_papers = [{
            "title": "Sample",
            "abstract": "Sample abstract",
            "section_names": ["Introduction", "Methodology", "Results", "Discussion"],
            "sections": ["Sample intro", "Sample method", "Sample results", "Sample discussion"]
        }]
    papers_dataset = Dataset.from_list(train_papers)
    logger.info(f"Loaded {len(train_papers)} ArXiv papers")

    # Load CSV data
    csv_texts = []
    if os.path.exists("sample_data.csv"):
        try:
            data = pd.read_csv("sample_data.csv")
            csv_texts.append(data.to_string())
            logger.info("Loaded sample_data.csv for tokenizer training")
        except Exception as e:
            logger.error(f"Error reading sample_data.csv for tokenizer: {e}")

    # Train tokenizer
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[PAD]", "[EOS]", "[SEP]"], vocab_size=10000)
    valid_texts = (
        wikitext_texts +
        [f"Title: {row.get('title', '')}\nAbstract: {row.get('abstract', '')}\nSections: {' '.join(row.get('sections', []))}" for row in papers_dataset] +
        csv_texts
    )
    tokenizer.train_from_iterator(valid_texts, trainer)
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Trained tokenizer, vocab_size: {vocab_size}")
    tokenizer.save("research_tokenizer.json")

    # Cache tokenizer
    cache_file = os.path.join(data_dir, "tokenized_cache.pkl")
    if not os.path.exists(cache_file):
        with open(cache_file, "wb") as f:
            pickle.dump({"tokenizer": tokenizer, "vocab_size": vocab_size}, f)

    # Process papers
    def format_paper(row):
        data_summary = "No tabular data provided."
        if os.path.exists("sample_data.csv"):
            try:
                data = pd.read_csv("sample_data.csv")
                required_columns = ["accuracy", "f1_score", "training_time_hours"]
                if all(col in data.columns for col in required_columns):
                    data_summary = (
                        f"Dataset contains {len(data)} models with metrics:\n"
                        f"Average accuracy: {data['accuracy'].mean():.2f}\n"
                        f"Average F1 score: {data['f1_score'].mean():.2f}\n"
                        f"Average training time: {data['training_time_hours'].mean():.2f} hours"
                    )
                else:
                    logger.error(f"Missing required columns in sample_data.csv: {required_columns}")
            except Exception as e:
                logger.error(f"Error reading sample_data.csv: {e}")
        section_names = row.get('section_names', ['Introduction', 'Methodology', 'Results', 'Discussion'])
        sections = row.get('sections', ['N/A'] * 4)
        return (
            f"[SEP]Title: {row.get('title', 'Untitled')}\n"
            f"[SEP]Abstract: {row.get('abstract', 'N/A')}\n"
            f"[SEP]Data: {data_summary}\n"
            f"[SEP]Introduction: {sections[0] if len(sections) > 0 else section_names[0]}\n"
            f"[SEP]Methodology: {sections[1] if len(sections) > 1 else section_names[1]}\n"
            f"[SEP]Results: {sections[2] if len(sections) > 2 else section_names[2]}\n"
            f"[SEP]Discussion: {sections[3] if len(sections) > 3 else section_names[3]}\n[EOS]"
        )

    papers = [format_paper(row) for row in papers_dataset]
    return tokenizer, wikitext_texts, papers, vocab_size

def create_dataloader(texts, tokenizer, batch_size=4, max_length=2048):
    valid_texts = [text for text in texts if isinstance(text, str) and text.strip()]
    if not valid_texts:
        logger.error("No valid texts provided for tokenization")
        valid_texts = ["Sample text for tokenization"]
    logger.info(f"Processing {len(valid_texts)} valid texts out of {len(texts)}")

    encodings = []
    for text in valid_texts:
        try:
            enc = tokenizer.encode(text).ids[:max_length]
            if enc:
                encodings.append(enc)
            else:
                logger.warning(f"Empty encoding for text: {text[:50]}...")
                encodings.append([tokenizer.token_to_id("[PAD]")])
        except Exception as e:
            logger.warning(f"Failed to encode text: {text[:50]}... Error: {e}")
            encodings.append([tokenizer.token_to_id("[PAD]")])

    if not encodings:
        logger.error("No valid encodings produced")
        raise ValueError("No valid encodings produced after tokenization")

    vocab_size = tokenizer.get_vocab_size()
    try:
        max_index = max(max(enc) for enc in encodings)
        logger.info(f"Max token index: {max_index}, vocab_size: {vocab_size}")
        if max_index >= vocab_size:
            logger.warning(f"Clamping indices: max_index {max_index} >= vocab_size {vocab_size}")
            encodings = [[min(token, vocab_size - 1) for token in enc] for enc in encodings]
    except ValueError as e:
        logger.error(f"Error computing max index: {e}")
        encodings = [[tokenizer.token_to_id("[PAD]")] for _ in valid_texts]

    encodings = [e + [tokenizer.token_to_id("[PAD]")] * (max_length - len(e)) for e in encodings]
    dataset = torch.tensor(encodings, dtype=torch.long)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)