# SDG 3 Multi-Label Text Classification using Machine Learning and Deep Learning

### TRADITIONAL CLASSICAL ML EXPERIMENTS

## Project Overview

This project investigates the use of Natural Language Processing (NLP), traditional machine learning, and deep learning techniques for the automatic classification of development-related documents into Sustainable Development Goal 3 (SDG 3) indicators.

The dataset consists of grants, tenders, contracts, organizational profiles, humanitarian initiatives, and development reports. Since a single document may address multiple SDG indicators simultaneously, the problem is formulated as a **multi-label text classification task**.

The project includes:

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Multi-label target encoding
* Feature engineering using TF-IDF and Word2Vec
* Traditional machine learning models
* Deep learning models (RNN, LSTM, GRU, Transformer)
* Model evaluation using Hamming Loss and other classification metrics
* Final prediction generation for unseen test data

The primary evaluation metric for this assignment is **Hamming Loss**, where lower values indicate better performance.


# Requirements

This project was developed and tested using **Google Colab**.

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

Open the provided notebook file:

```text
SDG3_Assignment2.ipynb
```

in Google Colab.

---

## Step 2: Enable GPU Runtime

For deep learning experiments, GPU acceleration is recommended.

In Google Colab:

1. Click **Runtime**
2. Select **Change Runtime Type**
3. Under **Hardware Accelerator**
   select:

```text
GPU
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
train.csv
test.csv
```

The training dataset contains:

* document information
* SDG indicator labels

The test dataset contains:

* document information only
* no target labels

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
train.csv
test.csv
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

### Data Loading

The training and test datasets are loaded into memory.

### Data Cleaning

The raw text is cleaned by:

* removing HTML tags
* removing URLs
* removing email addresses
* decoding HTML entities
* converting text to lowercase
* removing punctuation
* normalizing whitespace

<img width="895" height="199" alt="Screenshot 2026-06-06 at 20 23 36" src="https://github.com/user-attachments/assets/8c817a82-0e34-473b-936b-859b40c95979" />


### Exploratory Data Analysis

The notebook generates:

* label distributions
* document type distributions
* text length statistics
* multi-label analysis
* co-occurrence visualizations

<img width="580" height="455" alt="indicators per doc" src="https://github.com/user-attachments/assets/39428f85-14c4-4afe-8510-d3a5ed7da4a4" />


### Multi-Label Encoding

The SDG indicators are transformed into a binary multi-hot encoded representation suitable for multi-label classification.

<img width="379" height="175" alt="Screenshot 2026-06-06 at 20 28 34" src="https://github.com/user-attachments/assets/5b879259-d861-4779-971d-eeaaecb75d2a" />


### Feature Engineering

Two feature extraction strategies are evaluated:

#### TF-IDF

Documents are transformed into sparse TF-IDF vectors.

#### Word2Vec

Word embeddings are trained on the corpus and converted into document-level representations.

### Traditional Machine Learning Experiments

The notebook trains and evaluates:

* Logistic Regression
* Linear SVM
* Cosine Similarity Approaches

using both TF-IDF and Word2Vec representations.

### Model Evaluation

Performance metrics include:

* Hamming Loss
* Accuracy
* Precision
* Recall
* F1 Score

# Evaluation Metrics

The primary evaluation metric is:

## Hamming Loss

```text
Lower is Better
```

Hamming Loss measures the proportion of incorrectly predicted labels across all documents and SDG indicators.


<img width="747" height="636" alt="Screenshot 2026-06-07 at 10 09 50" src="https://github.com/user-attachments/assets/ebd3be37-9707-4a07-8d83-5b9985c43497" />


### Final Prediction Generation

The top three best performing models are:
- TF-IDF + Linear SVM =  Hamming Loss: 0.0450
- TF-IDF + Logistic Regression = Hamming Loss: 0.0532
- Word2Vec + Linear SVM = Hamming Loss : 0.0577

---

# Reproducibility

To ensure reproducibility:

* Use the provided notebook without modification.
* Enable GPU runtime before training deep learning models.
* Run notebook cells in order.
* Use the supplied training and test datasets.
* Keep random seeds unchanged where specified.

Following these steps should reproduce the preprocessing pipeline, experiments, evaluation metrics, and final predictions reported in the accompanying research report.

---

# Authors

**NAME: ** NTWARI MIKE CHRIS KEVIN

