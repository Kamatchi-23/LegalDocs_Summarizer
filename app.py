import torch
import evaluate
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
import streamlit as st
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from transformers import AdamW, get_scheduler
import evaluate
from dotenv import load_dotenv
import os

# Define paths to models 
# Load environment variables from .env file
load_dotenv()

# Define paths to models using environment variables or fallback to default
longt5_model_path = os.getenv("LONGT5_MODEL_PATH", "models/fine_tuned_long_t5_model")
gpt2_model_path = os.getenv("GPT2_MODEL_PATH", "models/fine_tuned_gpt2_model")
llama_model_path = os.getenv("LLAMA_MODEL_PATH", "models/fine_tuned_llama3_model")

# Load LongT5 model and tokenizer
longt5_tokenizer = AutoTokenizer.from_pretrained(longt5_model_path)
longt5_model = LongT5ForConditionalGeneration.from_pretrained(longt5_model_path)

# Load GPT-2 model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_path)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model.config.pad_token_id = gpt2_tokenizer.eos_token_id
# Load Llama3 model and tokenizer
llama_tokenizer = PreTrainedTokenizerFast.from_pretrained(llama_model_path)
llama_model = LlamaForCausalLM.from_pretrained(llama_model_path)
llama_tokenizer.pad_token = llama_tokenizer.eos_token


# Function to generate summary
def generate_summary(model, tokenizer, input_text, max_length=300):
    # Tokenizing the inputs
    inputs = "Summarize the following judgement: " + input_text
    model_inputs = tokenizer(inputs, max_length=512, padding='max_length', truncation=True, return_tensors="pt")
    
    if model== longt5_model:
        summary_ids = model.generate(input_ids=model_inputs["input_ids"], attention_mask=model_inputs["attention_mask"], max_length=max_length, num_beams=4, early_stopping=True)
    else:
        summary_ids = model.generate(input_ids=model_inputs["input_ids"], attention_mask=model_inputs["attention_mask"], max_new_tokens=300, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to compute ROUGE scores
def compute_rouge_scores(pred_summary, ref_summary):
    rouge = evaluate.load('rouge')
    scores = rouge.compute(predictions=[pred_summary], references=[ref_summary])
    return scores

# Streamlit App
st.title("Legal Text Summarization and ROUGE Score Evaluation")

# Input Text
input_text = st.text_area("Enter legal text to summarize:")

# Reference Text for ROUGE (optional)
reference_text = st.text_area("Enter reference summary (optional for ROUGE evaluation):", "")

# Choose Model
model_choice = st.selectbox("Choose model:", ["LongT5", "GPT-2", "Llama3.2-1B"])

# Summarize Button
if st.button("Generate Summary"):
    if input_text:
        if model_choice == "LongT5":
            summary = generate_summary(longt5_model, longt5_tokenizer, input_text)
        elif model_choice == "GPT-2":
            summary = generate_summary(gpt2_model, gpt2_tokenizer, input_text)
        else:
            summary = generate_summary(llama_model, llama_tokenizer, input_text)

        st.success(summary)

        if reference_text:
            rouge_scores = compute_rouge_scores(summary, reference_text)
            st.write("ROUGE Scores:")
            st.json(rouge_scores)
    else:
        st.error("Please input text to summarize.")