A PyTorch-based sentiment analysis model built completely from scratch to classify IMDB movie reviews as positive or negative[cite: 1, 2]. This project demonstrates the inner workings of Transformer architectures and natural language processing (NLP) pipelines.

## Project Overview
**SentimentScope** is a binary sentiment classifier framed as a recommendation system enhancement for an entertainment company (CineScope)[cite: 1, 2]. Instead of fine-tuning a pre-existing large language model, this project implements a custom GPT-style Transformer model (DemoGPT) entirely from scratch using PyTorch[cite: 1, 2].

## Key Features
* **Custom Transformer Architecture**: Implements fundamental transformer components from the ground up, including[cite: 1, 2]:
  * `AttentionHead`[cite: 1, 2]
  * `MultiHeadAttention`[cite: 1, 2]
  * `FeedForward` Networks[cite: 1, 2]
  * Transformer `Block` with Layer Normalization[cite: 1, 2]
* **Subword Tokenization**: Utilizes Hugging Face's `bert-base-uncased` tokenizer for robust text preprocessing[cite: 1, 2].
* **Interactive UI**: Includes an `ipywidgets`-powered interactive demonstration block at the end of the notebook to test custom movie reviews in real-time[cite: 2].

## Dataset
The model is trained on the Stanford IMDB Dataset, consisting of highly polar movie reviews[cite: 1, 2].
* **Training Set**: 22,500 reviews[cite: 1, 2]
* **Validation Set**: 2,500 reviews[cite: 1, 2]
* **Test Set**: 25,000 reviews[cite: 1, 2]
* **Max Sequence Length**: 128 tokens[cite: 1, 2]

## Results
The custom `DemoGPT` model was trained for 10 epochs using the AdamW optimizer and achieved the following performance[cite: 2]:
* **Validation Accuracy**: ~79.80%[cite: 2]
* **Test Accuracy**: 76.73% (Successfully surpassing the >75% project baseline)[cite: 2]

## Tech Stack
* **Language**: Python
* **Frameworks & Libraries**: PyTorch, Hugging Face `transformers`, Pandas, Matplotlib, Seaborn, ipywidgets[cite: 1, 2]
* **Environment**: Jupyter Notebook / Google Colab[cite: 1, 2]

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/Fa-Alsuqayri/MovieSentimentScope.git
