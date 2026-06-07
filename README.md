# SDG 3 Multi-Label Text Classification Using Machine Learning and Deep Learning

## Project Overview

This project investigates the use of Natural Language Processing (NLP), traditional machine learning, and deep learning techniques for the automatic classification of development-related documents into Sustainable Development Goal 3 (SDG 3) indicators.

The United Nations Sustainable Development Goals (SDGs) provide a global framework for addressing major social, economic, and environmental challenges. SDG 3 focuses on ensuring healthy lives and promoting well-being for all at all ages. Monitoring progress toward these indicators requires analyzing large volumes of textual information generated through grants, contracts, tenders, organizational reports, humanitarian initiatives, and development programs.

Manually categorizing such documents is time-consuming and difficult to scale. This project explores automated approaches capable of predicting the SDG 3 indicators addressed by a document using machine learning and deep learning methods.

The task is formulated as a **multi-label text classification problem**, since a single document may be associated with multiple SDG indicators simultaneously.

---

## Problem Statement

Given a development-related document, predict all relevant SDG 3 indicators associated with the document.

Unlike traditional classification problems where each document belongs to only one category, documents in this dataset may correspond to multiple indicators at the same time. Consequently, the objective is to predict a set of labels rather than a single class.

The primary evaluation metric for this project is:

### Hamming Loss

Lower Hamming Loss values indicate better classification performance.

Additional evaluation metrics used throughout experimentation include:

* Accuracy
* Precision (Macro)
* Recall (Macro)
* F1-Score (Macro)

---

## Dataset Description

The dataset consists of approximately 3,000 development-related documents collected from multiple sources including:

* Grants
* Contracts
* Tenders
* Organizational Profiles
* Development Initiatives
* Humanitarian Programs
* Reports

Two datasets were provided:

### Training Dataset

Contains:

* Unique ID
* Document Type
* Document Text
* SDG Indicator Labels

### Test Dataset

Contains:

* Unique ID
* Document Type
* Document Text

The test dataset does not contain target labels and is used for final prediction generation.

---

## Project Workflow

The overall workflow followed throughout the project is shown below:

```text
Raw Documents
      │
      ▼
Text Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Multi-Label Encoding
      │
      ▼
Feature Engineering
      │
      ├── TF-IDF
      │
      └── Word2Vec
      │
      ▼
Model Development
      │
      ├── Traditional Machine Learning
      ├── LSTM
      ├── GRU
      └── Transformer
      │
      ▼
Evaluation
      │
      ▼
Final Predictions
```

---

## Data Preprocessing

Before model development, the raw textual data underwent several preprocessing steps.

The preprocessing pipeline included:

* HTML entity decoding
* HTML tag removal
* URL removal
* Email address removal
* Text normalization
* Lowercase conversion
* Punctuation removal
* Whitespace normalization

The purpose of preprocessing was to reduce noise while preserving meaningful semantic content relevant to SDG classification.

<img width="895" height="199" alt="Screenshot 2026-06-06 at 20 23 36" src="https://github.com/user-attachments/assets/bf69d582-0629-4b7a-9749-a25efbe4022c" />


---

## Target Transformation

The original SDG labels were distributed across multiple columns.

Since a document may belong to multiple SDG indicators simultaneously, the problem was formulated as a multi-label classification task.

The target labels were therefore transformed into a multi-hot encoded representation where:

* 1 indicates the presence of an indicator
* 0 indicates the absence of an indicator

This representation was used throughout all experiments.

<img width="379" height="175" alt="Screenshot 2026-06-06 at 20 28 34" src="https://github.com/user-attachments/assets/568f5a64-b7e2-46bd-be19-270fabf43545" />


---

## Feature Engineering

Two feature extraction approaches were investigated.

### TF-IDF

TF-IDF representations were used to generate sparse document vectors based on term importance.

### Word2Vec

Word2Vec embeddings were trained on the corpus to learn semantic representations of words and documents.

These representations formed the basis for both traditional machine learning and deep learning experiments.

---

# Repository Structure

The repository is organized according to the experimental tracks conducted by different group members.

```text
.
├── README.md
│
├── Traditional Classical Model experimentation.ipynb
└── traditionzl_Models_Exp_README.md
│
├── LSTM_Experiments.ipynb
├── README.md
│

├── GRU_Experiments.ipynb
├── README.md
│
├── Transformer_Experiments.ipynb
├── README.md
│
└── Research_Report.pdf
```

---

# Experimental Contributions

The project was divided into multiple experimental tracks to investigate different modeling approaches.

## Traditional Machine Learning Models

**Researcher:** NTWARI MIKE CHRIS KEVIN

This experimental track focuses on classical machine learning approaches using TF-IDF and Word2Vec feature representations.

Models evaluated include:

* Logistic Regression
* Linear SVM
* Cosine Similarity-Based Classification

For implementation details, experiments, results, and discussion, refer to:

```text
traditionzl_Models_Exp_README.md
```

---

## LSTM Experiments

**Researcher:** Innocent

This experimental track investigates Long Short-Term Memory (LSTM) networks using Word2Vec embeddings for multi-label SDG classification.

Experiments include:

* Baseline LSTM
* Bidirectional LSTM
* Hyperparameter Optimization
* Threshold Tuning

For implementation details and results, refer to:

```text
notebooks/LSTM/README.md
```

---

## GRU Experiments

**Researcher:** Sheryl

This experimental track investigates Gated Recurrent Unit (GRU) architectures for SDG indicator prediction.

Experiments include:

* Baseline GRU
* Bidirectional GRU
* Hyperparameter Optimization
* Comparative Evaluation

For implementation details and results, refer to:

```text
notebooks/GRU/README.md
```

---

## Transformer Experiments

**Researcher:** Sharif

This experimental track investigates Transformer-based architectures for multi-label text classification.

Experiments include:

* Transformer Encoder Models
* Attention-Based Architectures
* Hyperparameter Optimization
* Comparative Evaluation

For implementation details and results, refer to:

```text
notebooks/Transformer/README.md
```

---

# Running the Project

## Step 1: Open the Desired Notebook

Navigate to the experimental track of interest and open the corresponding notebook in Google Colab.

---

## Step 2: Enable GPU Runtime

In Google Colab:

```text
Runtime
    └── Change Runtime Type
            └── GPU
```

GPU acceleration is recommended for all deep learning experiments.

---

## Step 3: Install Dependencies

Run the first notebook cell:

```python
!pip install gensim
```

---

## Step 4: Download NLTK Resources

Run the second notebook cell:

```python
import nltk

nltk.download('all')
```

---

## Step 5: Upload Dataset

Upload:

```text
train.csv
test.csv
```

to the Colab environment or mount Google Drive.

---

## Step 6: Run All Cells

After uploading the datasets:

```text
Runtime
    └── Run All
```

The notebook will automatically execute the preprocessing pipeline, feature engineering steps, model training procedures, evaluation workflow, and final prediction generation.

---

# Evaluation Methodology

All models were evaluated using a consistent validation framework.

The primary metric used throughout the project was:

### Hamming Loss

```text
Lower is Better
```

Additional evaluation metrics included:

* Accuracy
* Precision (Macro)
* Recall (Macro)
* F1 Score (Macro)

Where appropriate, experiments also included:

* Learning Curves
* Loss Curves
* Confusion Matrices
* Model Comparison Tables

---

# Research Report

A complete academic report documenting the methodology, experiments, results, discussion, limitations, and future work is included within the repository.

```text
Report/Research_Report.pdf
```

---

# Demo Video

The project demonstration video provides an overview of:

* Problem formulation
* Data preprocessing
* Feature engineering
* Experimental methodology
* Model evaluation
* Key findings and lessons learned

The video can be found in:

```text
Video_Demo/
```

---

# Team Members

| Team Member             | Contribution                        |
| ------------------------| ----------------------------------- |
| Ntwari Mike Chris Kevin | Traditional Machine Learning Models |
| Innocent                | LSTM Experiments                    |
| Sheryl                  | GRU Experiments                     |
| Sharif                  | Transformer Experiments             |

---

# Key Finding

One of the most significant findings of the project was that TF-IDF combined with Linear SVM established a very strong baseline for SDG classification. This result demonstrated that domain-specific keywords and lexical patterns are highly informative for identifying SDG indicators. The deep learning experiments were subsequently designed to investigate whether sequential and contextual representations could further improve upon this baseline.
