import pandas as pd

def generate_research_paper(model, tokenizer, data_path, title, max_length=500):
    data = None
    try:
        data = pd.read_csv(data_path)
        data_summary = (
            f"Dataset contains {len(data)} models with metrics:\n"
            f"Average accuracy: {data['accuracy'].mean():.2f}\n"
            f"Average F1 score: {data['f1_score'].mean():.2f}\n"
            f"Average training time: {data['training_time_hours'].mean():.2f} hours\n"
            f"Dataset size: {data['dataset_size'].iloc[0]}"
        )
    except:
        data_summary = "No tabular data provided."
    prompt = (
        f"Title: {title}\n"
        f"[SEP]Data Summary: {data_summary}\n"
        f"[SEP]Introduction: Write a research paper based on the provided data, including Abstract, Introduction, Methodology, Results, and Discussion sections."
    )
    tokens = tokenizer.encode(prompt).ids
    print(f"Prompt length: {len(tokens)}")
    paper = model.generate(tokenizer, prompt, max_length=max_length)
    return paper