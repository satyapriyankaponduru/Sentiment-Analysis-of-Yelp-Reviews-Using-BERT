# Sentiment Analysis of Yelp Reviews using BERT

## Project Overview
This project focuses on leveraging the advanced transformer model BERT (Bidirectional Encoder Representations from Transformers) for sentiment analysis of Yelp reviews. By fine-tuning the BERT model on a dataset of customer reviews, the goal is to achieve improved accuracy in sentiment classification compared to traditional machine learning approaches such as Decision Trees, Logistic Regression, and Naive Bayes.

## Key Features
- **Advanced NLP Techniques**: Utilizes BERT, a state-of-the-art transformer model, for sentiment analysis.
- **Context-Aware Tokenization**: Preserves the contextual relationships in reviews for improved classification.
- **Optimized Fine-Tuning**: Implements optimization techniques to enhance model performance while avoiding overfitting.
- **Comprehensive Evaluation**: Employs metrics like Accuracy, F1 Score, and AUROC for robust performance evaluation.

## Steps Involved
### 1. Data Collection and Preparation
- Collected a balanced dataset of Yelp reviews, ensuring equal representation of positive and negative sentiments.
- Preprocessed the dataset by cleaning text, removing unnecessary characters, and standardizing formats.

### 2. Tokenization
- Tokenized reviews using BERT's tokenizer to preserve contextual relationships.
- Converted tokenized text into input IDs and attention masks suitable for the BERT model.

### 3. Fine-Tuning the BERT Model
- Loaded the pre-trained BERT model and added a classification layer.
- Fine-tuned the model using optimization techniques like learning rate scheduling and dropout to prevent overfitting.

### 4. Performance Evaluation
- Evaluated the fine-tuned model using various metrics:
  - **Accuracy**: Overall correctness of predictions.
  - **F1 Score**: Balance between precision and recall.
  - **AUROC**: Ability to distinguish between positive and negative classes.

## Results
- BERT outperformed traditional models such as Decision Trees, Logistic Regression, and Naive Bayes in sentiment classification.
- The confusion matrices and ROC curves illustrate model performance across different evaluation criteria.

## Installation and Usage
### Prerequisites
Ensure the following dependencies are installed:
- Python 3.8 or later
- PyTorch
- Transformers library by Hugging Face
- Scikit-learn
- Pandas, NumPy, and Matplotlib

### Installation
Clone this repository:
```bash
git clone https://github.com/your-username/sentiment-analysis-yelp-bert.git
cd sentiment-analysis-yelp-bert
