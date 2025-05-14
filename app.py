import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import time
import pandas as pd
from io import BytesIO

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.load_state_dict(torch.load("bert_fake_review_model.pth", map_location=device))
model.to(device)
model.eval()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Inject dark theme CSS
with open("style_dark.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Remove footer
st.markdown("""
    <style>
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='title'>🕵️‍♀️ Fake Review Detector</h1>", unsafe_allow_html=True)

# ----------------- Option 1: Single Review -----------------
st.subheader("🔍 Enter a Single Review")
with st.form(key="review_form"):
    review = st.text_area("Type your product review here...", height=150, key="review_input", label_visibility="collapsed")
    submitted = st.form_submit_button("Detect")

if submitted and review.strip():
    with st.spinner("Analyzing..."):
        inputs = tokenizer(review, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        time.sleep(0.5)

        if pred == 1:
            st.markdown("<div class='genuine-msg'>✅ This review seems genuine!</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='fake-msg'>⚠️ This review might be fake.</div>", unsafe_allow_html=True)

# ----------------- Option 2: Bulk Reviews Upload -----------------
st.subheader("📁 Upload a File with Reviews")
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        if "review" not in df.columns:
            st.error("The file must contain a 'review' column.")
        else:
            st.info("Analyzing reviews, please wait...")
            reviews = df["review"].astype(str).tolist()
            results = []

            for review_text in reviews:
                inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()
                results.append("Genuine ✅" if pred == 1 else "Fake ⚠️")

            df["Prediction"] = results
            st.success("✅ All reviews analyzed!")

            st.dataframe(df[["review", "Prediction"]], use_container_width=True)

            # Download button
            def convert_df(df):
                return df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", convert_df(df), "review_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error processing file: {e}")
