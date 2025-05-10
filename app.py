import streamlit as st
import pandas as pd
import torch
from model import ResearchLLM
from tokenizers import Tokenizer
from generate_paper import generate_research_paper

st.title("Research Paper Generator")
vocab_size = 10000
model = ResearchLLM(vocab_size=vocab_size, max_length=2048)
model.load_state_dict(torch.load("research_llm.pt"))
model.eval()
tokenizer = Tokenizer.from_file("research_tokenizer.json")

title = st.text_input("Enter the paper title:")
uploaded_file = st.file_uploader("Upload a CSV file with data", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data_path = "temp_data.csv"
    data.to_csv(data_path, index=False)
    if st.button("Generate Paper"):
        with st.spinner("Generating..."):
            paper = generate_research_paper(model, tokenizer, data_path, title)
            st.subheader("Generated Paper")
            st.write(paper)
else:
    st.warning("Please upload a CSV file.")