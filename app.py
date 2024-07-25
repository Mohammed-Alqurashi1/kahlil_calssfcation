import streamlit as st
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

# Load the trained model and tokenizer
model_name = r"C:\Users\moham\Desktop\project3\trained_model_seq2seq"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Function to clean and normalize text (same as used during training)
def clean_text(text):
    import re
    import string
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = text.strip()
    return text

# Streamlit app
st.title('Arabic Text Generation Based on Title')
st.write('Enter an Arabic title to generate content using the trained model.')

input_title = st.text_area("Enter the title here:", "")

if st.button('Generate Content'):
    if input_title:
        # Clean the input title
        cleaned_title = clean_text(input_title)

        # Tokenize the input title
        inputs = tokenizer(cleaned_title, padding='max_length', truncation=True, max_length=512, return_tensors="pt")

        # Generate content with logits
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], output_scores=True, return_dict_in_generate=True)

        # Decode the generated content
        generated_contents = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs.sequences]

        # Display the results
        for generated_content in generated_contents:
            st.write(f"**Title:** {input_title}")
            st.write(f"**Generated Content:** {generated_content}")
    else:
        st.warning("Please enter a title to generate content.")
