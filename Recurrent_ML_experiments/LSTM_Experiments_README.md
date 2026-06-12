# SDG 3 Multi-Label Text Classification using Machine Learning and Deep Learning

### LSTM DEEP LEARNING EXPERIMENTS

<a href="https://colab.research.google.com/github/Sharif2138/ML-techniques-1_Group-10_Formative-2/blob/main/LSTM_Model_Experiments.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Project Overview

This project investigates the use of Natural Language Processing (NLP), traditional machine learning, and deep learning techniques for the automatic classification of development-related documents into Sustainable Development Goal 3 (SDG 3) indicators.

The dataset consists of grants, tenders, contracts, organizational profiles, humanitarian initiatives, and development reports. Since a single document may address multiple SDG indicators simultaneously, the problem is formulated as a **multi-label text classification task**.

The project includes:

* Data loading and problem understanding
* Exploratory Data Analysis (EDA)
* Text cleaning and preprocessing
* Multi-label target encoding
* Feature engineering using TF-IDF and Word2Vec
* Deep learning models using LSTM architectures
* Model evaluation using Hamming Loss and other classification metrics
* Threshold tuning for optimal prediction
* Final prediction generation for unseen test data

The primary evaluation metric for this assignment is **Hamming Loss**, where lower values indicate better performance.


# Requirements

This project was developed and tested using **Google Colab** with a T4 GPU.

Required Python libraries include:

```python
tensorflow
numpy
pandas
scikit-learn
matplotlib
seaborn
nltk
gensim
beautifulsoup4
```

Most dependencies are automatically installed within the notebook.

---

# Running the Project on Google Colab

## Step 1: Open the Notebook

<a href="https://colab.research.google.com/github/Sharif2138/ML-techniques-1_Group-10_Formative-2/blob/main/LSTM_Model_Experiments.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Open the provided notebook file:

```text
LSTM_Model_Experiments.ipynb
```

in Google Colab.

---

## Step 2: Enable GPU Runtime

GPU acceleration is required for LSTM training.

In Google Colab:

1. Click **Runtime**
2. Select **Change Runtime Type**
3. Under **Hardware Accelerator**
   select:

```text
GPU (T4)
```

4. Click **Save**

To verify GPU availability:

```python
import tensorflow as tf

print("GPU Available:",
      tf.config.list_physical_devices('GPU'))
```

---

## Step 3: Install Required Packages

Run the first notebook cell:

```python
!pip install gensim
```

Wait until installation completes successfully.

---

## Step 4: Download NLTK Resources

Run the second notebook cell:

```python
import nltk

nltk.download('all')
```

This step downloads all required NLTK corpora, tokenizers, stopword lists, and linguistic resources used during preprocessing and feature engineering.

The download may take several minutes when executed for the first time.

---

## Step 5: Obtain the Dataset

Download the following datasets provided for the assignment:

```text
Devex_train.csv
Devex_test_questions.csv
```

The training dataset contains:

* Unique ID
* Document Type
* Document Text
* SDG indicator labels (Label 1 through Label 12)

The training dataset has **2,995 documents** across **8 document types**.

The test dataset contains:

* Unique ID
* Document Type
* Document Text

The test dataset has **998 documents** and contains no target labels.

---

## Step 6: Upload the Dataset to Google Colab

Upload both datasets into the Colab session.

You may use either:

### Method 1: Manual Upload

Run:

```python
from google.colab import files

uploaded = files.upload()
```

Then select:

```text
Devex_train.csv
Devex_test_questions.csv
```

from your computer.

### Method 2: Google Drive

Mount your Google Drive:

```python
from google.colab import drive

drive.mount('/content/drive')
```

and update the dataset paths accordingly.

---

## Step 7: Run All Notebook Cells

After the datasets have been uploaded:

1. Click **Runtime**
2. Select:

```text
Run All
```

or execute cells sequentially from top to bottom.

The notebook performs the following workflow:

### Step 1: Data Loading and Problem Understanding

The training and test datasets are loaded into memory and inspected.

* Understanding the task (multi-label classification)
* Understanding the variables — X (text) and Y (SDG indicator labels)
* Examining the data structure

The training data has 15 columns including Unique ID, Type, Text, and Label 1 through Label 12. Most documents have 1 or 2 SDG indicators assigned.

### Step 2: Exploratory Data Analysis

The notebook:

* Identifies label columns (Label 1 to Label 12)
* Creates a `num_indicators` column counting how many indicators each document has
* Creates an `indicators` column combining all label values per document
* Creates an `indicator_codes` column extracting just the short codes (e.g. `3.b.2, 3.c.1`)
* Identifies **27 unique SDG indicators** across the dataset

Most documents are assigned 1 indicator, followed by 2, then 3.

### Step 3 (Part A): Text Cleaning

The raw text is cleaned using a dedicated pipeline:

```python
def clean_text(text):
    # Decode HTML entities
    # Remove HTML tags using BeautifulSoup
    # Remove URLs
    # Convert to lowercase
    # Remove email addresses
    # Remove punctuation
    # Normalize whitespace
```

* Numerical values are preserved as they carry meaningful health-related information.

### Step 3 (Part B): Multi-Label Encoding

The SDG indicators are transformed into a binary multi-hot encoded representation using `MultiLabelBinarizer`.

Since a document can belong to multiple indicators simultaneously:

```text
Doc 1 belongs to [3.b.2, 3.c.1]  → [1, 1, 0, 0, ...]
Doc 2 belongs to [3.8.1, 3.4.1]  → [0, 0, 1, 1, ...]
```

### Step 3: Text Vectorization

Two representations are explored:

#### Pipeline 1 — TF-IDF

Documents are transformed into sparse TF-IDF vectors. Because TF-IDF performs its own tokenization, cleaned tokens are re-joined back into strings before vectorization.

#### Pipeline 2 — Word Embeddings (Word2Vec)

Word2Vec embeddings are trained on the corpus with the following configuration:

```python
Word2Vec(
    sentences=sentences,
    vector_size=200,
    window=5,
    min_count=2,
    workers=4
)
```

Document-level vectors are created by averaging all word vectors in each document.

### Step 4: Deep Learning Feature Preparation

For the LSTM models, a Keras `Tokenizer` is used to convert text to integer sequences, then padded to a fixed length:

```python
max_words = 50000
max_len   = 200
```

A Word2Vec embedding matrix (vector size = 100) is built from the training vocabulary and loaded into a Keras `Embedding` layer with `trainable=True`.

---

## Step 8: LSTM Model Experiments

### Experiment 1 — Baseline Bidirectional LSTM

The baseline model uses a single Bidirectional LSTM layer:

```python
inputs → Embedding → Bidirectional(LSTM(128)) → Dense(27, sigmoid)
```

Training configuration:

* Optimizer: Adam (learning rate = 1e-3)
* Loss: Binary Crossentropy
* Epochs: 10
* Batch Size: 32

### Experiment 2 — Improved Bidirectional LSTM

The improved model uses a deeper stacked architecture with pooling and regularization:

```python
inputs
  → Embedding
  → Bidirectional(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
  → Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
  → GlobalAveragePooling1D + GlobalMaxPooling1D  (concatenated)
  → Dense(256, relu) → Dropout(0.4)
  → Dense(128, relu) → Dropout(0.3)
  → Dense(27, sigmoid)
```

Training configuration:

* Optimizer: Adam (learning rate = 2e-4)
* Loss: Binary Crossentropy
* Epochs: 10
* Batch Size: 32

---

## Step 9: Evaluation

Model performance is evaluated using:

1. Training and validation loss curves
2. Training and validation accuracy curves
3. Classification metrics
4. Confusion matrix per label
5. Threshold tuning

### Training Curves

Loss and accuracy curves are plotted for both experiments to monitor learning progress and detect overfitting.

### Threshold Tuning

Instead of using a fixed threshold of 0.5, the optimal decision threshold is found by searching:

```python
thresholds = np.arange(0.1, 0.9, 0.05)
```

The threshold that minimizes Hamming Loss on the validation set is selected.

---

# Evaluation Metrics

The primary evaluation metric is:

## Hamming Loss

```text
Lower is Better
```

Hamming Loss measures the proportion of incorrectly predicted labels across all documents and SDG indicators.

---

# Experiment Results

### Experiment 1 — Baseline Bidirectional LSTM

| Metric | Value |
|---|---|
| Hamming Loss (threshold=0.5) | 0.0591 |
| Best Threshold | 0.65 |
| Best Hamming Loss (after tuning) | 0.0571 |
| F1 Score (macro) | 0.3484 |
| Precision (macro) | 0.5101 |
| Recall (macro) | 0.2837 |
| Accuracy | 0.2588 |

### Experiment 2 — Improved Bidirectional LSTM

| Metric | Value |
|---|---|
| Hamming Loss (threshold=0.5) | 0.0606 |
| Best Threshold | 0.40 |
| Best Hamming Loss (after tuning) | 0.0602 |
| F1 Score (macro) | 0.1348 |
| Precision (macro) | 0.2408 |
| Recall (macro) | 0.1051 |
| Accuracy | 0.2037 |

### Summary

The Baseline Bidirectional LSTM (Experiment 1) outperformed the deeper Improved LSTM (Experiment 2) across all metrics. The best performing model from this experimental track is:

* **Baseline Bidirectional LSTM** — Hamming Loss: **0.0571** (after threshold tuning to 0.65)

---

# Reproducibility

To ensure reproducibility:

* Use the provided notebook without modification.
* Enable GPU runtime before training deep learning models.
* Run notebook cells in order from top to bottom.
* Use the supplied training and test datasets.
* Keep random seeds unchanged where specified.

Following these steps should reproduce the preprocessing pipeline, experiments, evaluation metrics, and final predictions reported in the accompanying research report.

---

# Authors

**NAME:** Innocent Nangah
