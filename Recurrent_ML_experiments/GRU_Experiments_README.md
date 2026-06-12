Here it is:

---

# SDG 3 Indicator Classification — GRU Model Experiments

<a href="https://colab.research.google.com/github/Sharif2138/ML-techniques-1_Group-10_Formative-2/blob/main/Recurrent_ML_experiments/GRU_Model_Experiments_.ipynb" target="_parent">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Overview

This notebook contains the GRU (Gated Recurrent Unit) experiments for the SDG 3 multi-label text classification project. The goal of the project is to build a machine learning system that reads a text document and predicts which of the 27 SDG 3 health indicators are relevant to it.

Three GRU experiments were conducted, each building on the previous one with a targeted architectural improvement. Unlike classical models like TF-IDF which treat text as a bag of words, GRU networks read text sequentially — word by word — and maintain a memory of context. This makes them better suited for understanding the meaning of health and development phrases that depend on word order.

---

## Dataset

|             | Train                                                     | Test |
| ----------- | --------------------------------------------------------- | ---- |
| Documents   | 2,995                                                     | 998  |
| Labels      | 27 SDG 3 indicators                                       | —    |
| Text source | Tenders, reports, news articles, humanitarian initiatives | —    |

The 27 labels are SDG 3 indicator codes ranging from 3.1.1 to 3.d.1. Each document can belong to one or more indicators making this a multi-label classification problem.

---

## Requirements

pip install torch scikit-learn pandas numpy matplotlib seaborn nltk

---

## How to Run

1. Open the notebook in Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Upload Devex_train.csv and Devex_test_questions.csv when prompted
4. Run all cells from top to bottom

---

## Notebook Structure

| Cell                         | Description                                                                          |
| ---------------------------- | ------------------------------------------------------------------------------------ |
| Install libraries            | Installs PyTorch and all required dependencies                                       |
| Import libraries             | Imports all packages and sets random seeds for reproducibility                       |
| Load data                    | Loads training and test CSV files                                                    |
| Label detection and Y matrix | Extracts 27 SDG indicator codes and builds a binary label matrix of shape 2995 by 27 |
| Text cleaning                | Removes HTML, URLs, special characters and stopwords                                 |
| Build vocabulary             | Builds a 30,000 word vocabulary and encodes documents as integer sequences           |
| Dataset and DataLoader       | Creates PyTorch datasets with 256-token padding and an 80/20 train/val split         |
| Training helpers             | Defines training loop, evaluation function, threshold tuning and loss curve plotting |
| Experiment 1                 | Basic GRU — unidirectional, single layer                                             |
| Experiment 2                 | Bidirectional GRU — reads text in both directions                                    |
| Experiment 3                 | BiGRU + Self-Attention — focuses on the most relevant token positions                |
| Results summary              | Table and bar chart comparing all three experiments                                  |
| Per-label heatmap            | F1 score for each of the 27 indicators across all experiments                        |
| Inference                    | Generates test predictions using the best model with tuned thresholds                |
| Save weights                 | Downloads trained model weights as .pt files                                         |

---

## Experiments

### Experiment 1 — Basic GRU

A single-layer unidirectional GRU that reads the document left to right and uses the final hidden state to make predictions. This is the GRU baseline.

Embedding dim: 128
Hidden dim: 128
Dropout: 0.3
Direction: Unidirectional

| Metric       | Result |
| ------------ | ------ |
| Hamming Loss | 0.0607 |
| Macro F1     | 0.11   |

---

### Experiment 2 — Bidirectional GRU

Two GRUs run simultaneously — one forward, one backward. Their final hidden states are concatenated giving a 256-dimensional document representation. This allows the model to see context from both directions.

Embedding dim: 128
Hidden dim: 128 (256 after concatenation)
Dropout: 0.3
Direction: Bidirectional

| Metric       | Result |
| ------------ | ------ |
| Hamming Loss | 0.0512 |
| Macro F1     | 0.34   |

---

### Experiment 3 — Bidirectional GRU + Self-Attention (Best Model)

A self-attention layer is added on top of the BiGRU. Instead of only using the final hidden state, attention scores all 256 token positions and builds a weighted average of the most relevant ones. This is especially useful for long documents where key phrases can appear anywhere.

Embedding dim: 128
Hidden dim: 128 (256 after concatenation)
Dropout: 0.3
Direction: Bidirectional
Attention: Additive self-attention

| Metric            | Result |
| ----------------- | ------ |
| Hamming Loss      | 0.0483 |
| Precision (Macro) | 0.66   |
| Recall (Macro)    | 0.35   |
| Macro F1          | 0.42   |

---

## Results Summary

| Experiment   | Model                  | Hamming Loss | Macro F1 |
| ------------ | ---------------------- | ------------ | -------- |
| Experiment 1 | Basic GRU              | 0.0607       | 0.11     |
| Experiment 2 | Bidirectional GRU      | 0.0512       | 0.34     |
| Experiment 3 | BiGRU + Self-Attention | 0.0483       | 0.42     |

Experiment 3 achieved the lowest Hamming Loss of all 8 experiments in the group study at 0.0483.

---

## Key Design Decisions

Why PyTorch?
PyTorch makes the training loop fully explicit and visible. Every step — forward pass, loss computation, backpropagation, weight update — is written out clearly. This makes it easier to understand, explain and debug compared to frameworks like Keras where everything is hidden behind .fit().

Why learned embeddings instead of pre-trained ones?
A trainable embedding layer was used, initialized randomly and optimized jointly with the GRU during training. This keeps the approach self-contained and avoids external dependencies while still producing a meaningful word representation.

Why per-label threshold tuning?
Rather than applying a fixed 0.5 threshold to all 27 labels, the threshold for each label is tuned individually on the validation set to maximize that label's F1 score. This is important because rare labels need a lower threshold to be predicted at all.

Why early stopping?
Training stops automatically when the validation loss does not improve for 3 consecutive epochs. This prevents overfitting without requiring a fixed number of epochs.

---

## Limitations

- The dataset has only around 3,000 samples across 27 labels. Some indicators appear fewer than 30 times making them very difficult to learn.
- Recall is low at 0.35 across all models indicating conservative predictions — the model misses many relevant indicators.
- Neural networks generally need more training data than classical models. The lower Macro F1 compared to TF-IDF + SVM is expected at this dataset size.

---

## Output Files

| File                        | Description                                   |
| --------------------------- | --------------------------------------------- |
| gru_submission.csv          | Test set predictions — 998 rows by 28 columns |
| model_01_basic_gru.pt       | Saved weights for Experiment 1                |
| model_02_bigru.pt           | Saved weights for Experiment 2                |
| model_03_bigru_attention.pt | Saved weights for Experiment 3                |

---

## Author

This notebook covers the GRU model experiments as part of the SDG 3 Indicator Text Classification group assignment.
