A PyTorch-based sentiment analysis model built completely from scratch to classify IMDB movie reviews as positive or negative. This project demonstrates the inner workings of Transformer architectures and natural language processing (NLP) pipelines.

## Project Overview
**SentimentScope** is a binary sentiment classifier framed as a recommendation system enhancement for an entertainment company (CineScope). Instead of fine-tuning a pre-existing large language model, this project implements a custom GPT-style Transformer model (DemoGPT) entirely from scratch using PyTorch.

## Key Features
* **Custom Transformer Architecture**: Implements fundamental transformer components from the ground up, including:
  * `AttentionHead`
  * `MultiHeadAttention`
  * `FeedForward` Networks
  * Transformer `Block` with Layer Normalization
* **Subword Tokenization**: Utilizes Hugging Face's `bert-base-uncased` tokenizer for robust text preprocessing.
* **Interactive UI**: Includes an `ipywidgets`-powered interactive demonstration block at the end of the notebook to test custom movie reviews in real-time.

## Dataset
The model is trained on the Stanford IMDB Dataset, consisting of highly polar movie reviews.
* **Training Set**: 22,500 reviews
* **Validation Set**: 2,500 reviews
* **Test Set**: 25,000 reviews
* **Max Sequence Length**: 128 tokens

## Results
The custom `DemoGPT` model was trained for 10 epochs using the AdamW optimizer and achieved the following performance:
* **Validation Accuracy**: ~79.80%
* **Test Accuracy**: 76.73% (Successfully surpassing the >75% project baseline)

## Tech Stack
* **Language**: Python
* **Frameworks & Libraries**: PyTorch, Hugging Face `transformers`, Pandas, Matplotlib, Seaborn, ipywidgets
* **Environment**: Jupyter Notebook / Google Colab

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/Fa-Alsuqayri/MovieSentimentScope.git
