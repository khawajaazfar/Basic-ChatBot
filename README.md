# Simple PyTorch Chatbot 🤖💬

This repository contains a simple neural network-based chatbot built using PyTorch. The chatbot is designed to understand user intents from predefined patterns and provide appropriate responses.

## Table of Contents

* [Introduction](#introduction)
* [How it Works](#how-it-works)
* [Files in this Repository](#files-in-this-repository)
* [Requirements](#requirements)
* [How to Run](#how-to-run)
* [Training the Model](#training-the-model)
* [Interacting with the Chatbot](#interacting-with-the-chatbot)
* [Contribution](#contribution)
* [License](#license)

## Introduction

This project implements a basic conversational AI agent (chatbot) using a feed-forward neural network in PyTorch. The chatbot's responses are based on recognizing user intents, which are defined in a JSON file. This allows for a flexible and extensible way to manage conversational flows for various domains, such as customer service, information retrieval, or simple Q&A.

## How it Works

The chatbot operates in two main phases:

1.  **Training**: A neural network is trained on a dataset of patterns and their corresponding tags (intents) defined in `intents.json`. During training, input sentences are tokenized, stemmed, and converted into a "bag-of-words" representation. The neural network learns to map these bag-of-words inputs to the correct intent tags.
2.  **Inference (Chatting)**: When a user types a sentence, it goes through the same preprocessing steps (tokenization, stemming, bag-of-words conversion). This processed input is then fed to the trained neural network. The network outputs probabilities for each intent, and the intent with the highest probability is selected. The chatbot then randomly picks a response associated with that predicted intent.

## Files in this Repository

* `intents.json`: This JSON file defines the conversational structure. It contains a list of intents, where each intent has:
    * `tag`: A unique identifier for the intent (e.g., "greeting", "goodbye", "payments").
    * `patterns`: A list of example phrases or sentences that fall under this intent.
    * `responses`: A list of possible responses the chatbot can give when this intent is detected.
* `nltk_utils.py`: Contains utility functions for natural language processing (NLP) tasks:
    * `tokenize(sentence)`: Splits a sentence into individual words or tokens.
    * `stem(word)`: Reduces a word to its root form (e.g., "organizing" -> "organ").
    * `bag_of_words(tokenized_sentence, words)`: Converts a tokenized sentence into a bag-of-words array, indicating the presence or absence of words from a predefined vocabulary.
* `model.py`: Defines the neural network architecture using PyTorch's `nn.Module`. It's a simple feed-forward neural network with three linear layers and ReLU activation functions.
* `train.py`: The script responsible for training the chatbot model:
    * Loads data from `intents.json`.
    * Tokenizes, stems, and prepares the training data (patterns and tags).
    * Defines hyperparameters like batch size, learning rate, and number of epochs.
    * Initializes the `NeuralNet` model.
    * Uses `CrossEntropyLoss` as the criterion and `Adam` as the optimizer.
    * Trains the model using a DataLoader, saving the trained model's state and other essential data (`input_size`, `hidden_size`, `output_size`, `all_words`, `tags`) to `data.pth`.
* `chat.py`: The script to interact with the trained chatbot:
    * Loads the trained model and other necessary data from `data.pth`. [cite: 1]
    * Tokenizes and converts user input into a bag-of-words. [cite: 1]
    * Feeds the processed input to the model to get a prediction. [cite: 1]
    * Selects a random response from the detected intent's responses. [cite: 1]
    * Allows interactive chatting via the console. [cite: 1]
* `data.pth`: (Generated after running `train.py`) This file stores the trained model's state dictionary and other metadata (like `all_words` and `tags`) required for inference in `chat.py`.

## Requirements

You'll need the following Python libraries installed:

* `torch`
* `numpy`
* `nltk` (for `nltk_utils.py`)

You can install them using pip:

```bash
pip install torch numpy nltk
