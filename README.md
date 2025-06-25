# Multilingual Sentiment Classification using BERT & LLMs

This project focuses on multilingual sentiment analysis using the [Multilingual Amazon Reviews Corpus (MARC)](https://huggingface.co/datasets/mteb/amazon_reviews_multi). It explores both transformer-based and large language model (LLM) approaches for classifying product reviews into 5 sentiment categories across English, French, German, and Spanish.

---

## 🚀 Project Overview

- Built a multilingual text classification pipeline using `bert-base-multilingual-cased` (mBERT) and fine-tuned it on Amazon reviews.
- Evaluated both supervised and zero-shot cross-lingual performance.
- Implemented a generative classification system using the Mistral-7B-Instruct model with prompt engineering.
- Tracked experiments and metrics using Weights & Biases (W&B).

---

## 🗂 Dataset

**Multilingual Amazon Reviews Corpus (MARC)**  
- 6 languages (subset used: English, French, German, Spanish)  
- Reviews mapped to labels: 0 (Very Negative) to 4 (Very Positive)  
- Balanced: 200,000 training samples/language  
- Source: [Hugging Face Dataset](https://huggingface.co/datasets/mteb/amazon_reviews_multi)

---

## 🧠 Models & Methods

### 🔹 BERT-based Fine-Tuning
- Model: `bert-base-multilingual-cased`
- Fine-tuned on English reviews for classification
- Evaluated using Accuracy and MAE
- Cross-lingual: trained on non-English (FR, DE, ES), tested on English

### 🔹 LLM-based Zero-Shot Classification
- Model: `Mistral-7B-Instruct-v0.2`
- Task-specific prompt design for sentiment labeling
- No fine-tuning required – pure prompt-based inference
- Quantized with BitsAndBytes for memory-efficient deployment

---

## 📊 Results

| Setup                        | Accuracy |
|-----------------------------|----------|
| Fine-tuned mBERT (EN)       | 63.6%    |
| Cross-lingual mBERT (FR/DE/ES → EN) | 58.0%    |
| Zero-shot Mistral-7B        | *Varied by review*, prompt-based |

---

## 🛠 Key Skills & Technologies

- `Transformers` (Hugging Face)
- `Datasets`, `Evaluate`
- `PyTorch`, `Trainer API`
- `BitsAndBytes` 4-bit quantization
- `Prompt Engineering`, `LLMs`
- `W&B` for tracking experiments
- `scikit-learn`, `Pandas`, `Matplotlib`

---
