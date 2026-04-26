import streamlit as st
import pandas as pd
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# LOAD MODEL
# =========================
model_path = "aubmindlab/bert-base-arabertv2"


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =========================
# ASPECTS
# =========================
aspects = ["food", "service", "price", "cleanliness", "delivery"]
label_map = {0: "negative", 1: "neutral", 2: "positive"}

# =========================
# UI
# =========================
st.title("🔍 ABSA Sentiment Analysis App")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file:

    df = pd.read_excel(uploaded_file)
    st.write("📄 Data Preview:", df.head())

    rows = []

    # =========================
    # BUILD INPUTS
    # =========================
    for _, row in df.iterrows():
        review_id = row["review_id"]
        review_text = str(row["review_text"])

        for aspect in aspects:
            rows.append({
                "review_id": review_id,
                "review_text": review_text,
                "aspect": aspect,
                "input": review_text + " [SEP] " + aspect
            })

    pred_df = pd.DataFrame(rows)

    # =========================
    # TOKENIZE + PREDICT
    # =========================
    encodings = tokenizer(
        list(pred_df["input"]),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"]
        )

    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

    pred_df["prediction"] = [label_map[p] for p in preds]

    # =========================
    # BUILD JSON
    # =========================
    final_output = {}

    for _, row in pred_df.iterrows():
        rid = int(row["review_id"])
        aspect = row["aspect"]
        sentiment = row["prediction"]

        if rid not in final_output:
            final_output[rid] = {}

        final_output[rid][aspect] = sentiment

    st.success("✅ Prediction Done!")

    st.json(final_output)

    # =========================
    # DOWNLOAD BUTTON
    # =========================
    json_str = json.dumps(final_output, ensure_ascii=False, indent=2)

    st.download_button(
        label="Download JSON",
        data=json_str,
        file_name="submission.json",
        mime="application/json"
    )