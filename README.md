# Disaster-Tweet-NLP-Classification
Social media provides real-time crisis updates, but not all disaster-related tweets are real. This project uses Natural Language Processing and machine learning to classify over 10,000 hand-labeled tweets as actual disasters or not.


### Problem Statement
Tweets may contain words like “fire”, “crash”, or “explosion” that can either indicate a real disaster or be used metaphorically. The task is to build a binary classifier that distinguishes actual disaster tweets from figurative or unrelated ones.

### 🗂️ Dataset Overview
- Train Set: 7,613 tweets
- Test Set: 3,263 tweets
- Max Length: 34 tokens per tweet
- Vocabulary Size: ~14,000
- Class Distribution: Slight imbalance

### Data Preprocessing
- Contraction Expansion (e.g. "ain’t" → "is not")
- Punctuation, URL, Mention, and Hashtag Removal
- Lowercasing & Whitespace Reduction
- Stopword Removal via nltk
- Tokenization using word_tokenize

### Exploratory Data Analysis (EDA)
- Word Clouds for disaster vs non-disaster tweets
- Top keywords and locations (e.g., "Typhoon", "Bombing", "Fire")
- Tweet length analysis (character and word distribution)
- Keyword presence correlation with target label

### Model Architecture

✅ Final RNN Model: Stacked Bi-directional GRU + Attention (Implemented in TensorFlow/Keras with GloVe embeddings)

- Embeddings: GloVe Twitter-200 (200D)
- RNN Layers: Bi-GRU (128 → 64 units)
- Attention Layer: Focuses on key contextual words in each tweet
- Dense Layers: 32 → 16 units with LeakyReLU, Dropout(0.5)
- Regularization: Batch Normalization
- Training: Adam optimizer, Binary Crossentropy loss
- Callbacks: EarlyStopping, ReduceLROnPlateau
- Evaluation: F1 Score, Accuracy, Confusion Matrix

🤖 Transformer Models (HuggingFace + TensorFlow)

1. DistilBERT (Lightweight BERT variant) - ~40% smaller and 60% faster than base BERT

- Retains 97% of performance
- Fine-tuned for binary classification

2. RoBERTa (Robustly Optimized BERT Approach) - Trained on 160GB of data using dynamic masking

- Outperforms BERT in text classification tasks

3. deBERTa-v3 (Disentangled BERT with Enhanced Attention) - Separates content and positional embeddings for deeper understanding

- High accuracy and contextual awareness


4. BERT Preprocessor + Classifier (KerasNLP) - Fine-tuned using DistilBertClassifier from KerasNLP

- Effective in handling noisy, short-form text like tweet

### Libararies Used

- tensorflow, keras
- gensim, nltk, sklearn
- matplotlib, seaborn, wordcloud

### Training Strategy

- Stratified K-Fold Cross Validation for better generalization
- Model Averaging across 5 runs to stabilize predictions
- Class Weights used to handle imbalance
- EarlyStopping & ReduceLROnPlateau callbacks

### Key Takeaways

- Domain-specific embeddings (GloVe Twitter) outperform general ones
- Attention layers help improve both accuracy and interpretability
- Complex models require a balance between performance and training time
- Averaging predictions across runs mitigates randomness


