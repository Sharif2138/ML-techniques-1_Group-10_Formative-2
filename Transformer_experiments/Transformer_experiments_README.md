# Transformer Experiments for Multi-Label SDG Indicator Classification

## Project Overview
This project looks at how to use Transformer models (specifically BERT) and modern Natural Language Processing (NLP) to automatically sort development documents into different **Sustainable Development Goal (SDG)** indicators. 

The dataset includes text from grants, tenders, contracts, organizational profiles, humanitarian initiatives, and development reports. Because one document can talk about more than one SDG indicator at the same time, this is a **multi-label text classification** task.

Unlike traditional machine learning models that need manual features (like TF-IDF or Word2Vec), this project uses a pre-trained BERT model. This model learns the deep meaning of the text directly from the sentences.

### Main Goals of the Experiments:
1. **Context Length:** Compare how the model performs when reading up to 256 words (tokens) versus 512 words.
2. **Training Time (Epochs):** See the difference between training the model for 10 times (epochs) versus 15 times.
3. **Decision Thresholds:** Compare a simple static threshold (0.5) against an optimized **per-label adaptive threshold** system.

---

## Requirements & Environment
This project was created and tested using **Google Colab**. 

### Main Python Libraries Used:
* `torch` (PyTorch for deep learning)
* `transformers` (Hugging Face ecosystem for BERT)
* `scikit-learn` (for calculating performance metrics)
* `pandas` and `numpy` (for handling data tables and matrices)
* `matplotlib` and `seaborn` (for creating graphs)

---

## How to Run the Project on Google Colab

### Step 1: Open the Notebook
Open your notebook file in Google Colab:
`Transformer_experiments_notebook.ipynb`

### Step 2: Turn on the GPU Runtime
Transformer models need a lot of computing power. You must turn on the GPU before you start training.
1. In the top menu of Google Colab, click **Runtime**.
2. Select **Change runtime type**.
3. Under *Hardware accelerator*, choose **GPU**.
4. Click **Save**.

### Step 3: Upload Your Datasets
Upload your `Devex_train.csv` and `Devex_test_questions.csv` files into the Google Colab file section on the left side of the screen.

### Step 4: Run the Cells
Run the cells one by one from top to bottom, or click **Runtime** -> **Run all**.

---

## Project Workflow inside the Notebook

1. **Exploratory Data Analysis (EDA):** Checking the dataset structure, finding the most common SDG labels, and looking at the length of the text documents.
2. **Text Cleaning:** Preparing the raw text by formatting data strings and removing unwanted noise so the tokenizer can read it easily.
3. **Dataset Tokenization:** Using the pre-trained BERT tokenizer to split words, add padding, and create attention masks and binary target tensors.
4. **Fine-Tuning Loop:** Training the `bert-base-uncased` model using the `BCEWithLogitsLoss` function and the AdamW optimizer.
5. **Threshold Optimization:** Finding the best classification decision scores for each individual SDG indicator to improve final predictions.

---

## Experimental Settings & Results

We used **Hamming Loss** as our primary metric (lower values are better). We also monitored **F1-Micro**, **F1-Macro**, and **F1-Sample** scores to see how well the model predicts rare labels.

| Experiment ID | Model Setup | Max Length | Training Epochs | Threshold Method | Hamming Loss ↓ | F1-Micro ↑ | F1-Macro ↑ | F1-Sample ↑ |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **E1** | BERT-base-uncased | 256 | 10 | Global (t = 0.5) | 0.0813 | 0.5769 | 0.5486 | 0.6295 |
| **E2** | BERT-base-uncased | 512 | 10 | Global (t = 0.5) | 0.0886 | 0.5704 | 0.5323 | 0.6251 |
| **E3 (Best)** | BERT-base-uncased | 256 | 15 | **Per-Label Adaptive** | **0.0430** | **0.6417** | **0.6116** | **0.6509** |

### Important Discoveries:
* **Per-Label Threshold (E3):** Finding a unique decision score for each label was the best update. It cut the error rate (Hamming Loss) almost in half from `0.0813` to `0.0430` and gave the highest F1-scores.
* **Text Length (E2):** Increasing the text length to 512 did not help. It actually made the results slightly worse. This means the most important information for the SDG indicators is at the beginning of the text (within the first 256 words).
* **More training time:** Training for 10 epochs wasnt enough for the model to converge well. Training for 15 epochs with adaptive settings gave much better results.

---

## Reproducibility
To get the exact same results:
* Use the provided notebook without changing the code structure.
* Make sure the GPU runtime is turned on before you run the training cells.
* Run all notebook cells in order from top to bottom.
* Keep the random state seeds exactly as they are written in the code.

---

## Author
**Name:** SHARIF KIVIIRI