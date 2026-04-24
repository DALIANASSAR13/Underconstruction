# Arabic ABSA Model – Hackathon Submission

## 📌 Overview

This project implements an Aspect-Based Sentiment Analysis (ABSA) model for Arabic customer feedback.

The model predicts sentiment (positive, negative, neutral) for predefined aspects in each review.

---

## 📂 Project Structure

* `train_rewritten.ipynb` → Training + inference code
* `submission.json` → Final predictions
* `arabert-absa/` → Saved model weights
* `requirements.txt` → Dependencies

---

## ⚙️ Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1. Train Model

Run all cells in the notebook until:

```python
trainer.train()
```

---

### 2. Run Inference (Prediction)

Run the prediction section to generate:

```bash
DeepX_submission.json
```

---

## 🧠 Model Details

* Base model: AraBERT
* Task: Aspect-Based Sentiment Analysis
* Input format:

  ```
  review_text [SEP] aspect
  ```

---

## 📊 Output Format

```json
[
  {
    "review_id": 1,
    "aspects": ["food", "service"],
    "aspect_sentiments": {
      "food": "positive",
      "service": "negative"
    }
  }
]
```

---

## ⚠️ Notes

* Datasets are NOT included as per competition rules
* Model weights are included in `/arabert-absa/`

---

## 👩‍💻 Team
Yomna Saad

Dalia Nasser

Maryam Mahmoud
