import streamlit as st
from model import ResearchLLM
from tokenizers import Tokenizer
from generate_paper import generate_research_paper
import torch

st.title("Research Paper Generator")
model = ResearchLLM(vocab_size=10000)
model.load_state_dict(torch.load("research_llm.pt"))
model.eval()
tokenizer = Tokenizer.from_file("research_tokenizer.json")

input_type = st.radio("Input type:", ("Data File", "Manual"))
if input_type == "Data File":
    uploaded_file = st.file_uploader("Upload CSV dataset:", type="csv")
    title = st.text_input("Paper Title:", "Study on Input Data")
    abstract = st.text_area("Abstract:", "Provide a brief summary of the study.")
    if st.button("Generate Paper"):
        if uploaded_file:
            with st.spinner("Generating paper..."):
                import pandas as pd
                import io
                data = pd.read_csv(uploaded_file)
                data_path = "temp_data.csv"
                data.to_csv(data_path, index=False)
                paper = generate_research_paper(model, tokenizer, data_path, title, abstract)
                st.subheader("Generated Paper")
                st.write(paper)
        else:
            st.warning("Please upload a CSV file.")
else:
    title = st.text_input("Paper Title:", "Study on Input Data")
    abstract = st.text_area("Abstract:", "Provide a brief summary of the study.")
    data_desc = st.text_area("Data Description:", "Describe your data (e.g., table structure, key findings).")
    if st.button("Generate Paper"):
        if title and abstract:
            with st.spinner("Generating paper..."):
                paper = generate_research_paper(model, tokenizer, None, title, abstract)
                st.subheader("Generated Paper")
                st.write(paper)
        else:
            st.warning("Please provide a title and abstract.")