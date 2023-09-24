# Text Classification- NLP Project - Cyberbullying Detection and Product Classification - Binary and Multi Class Classification

#### Cyberbullying Detection - Notebook - Machine Learning Models: https://www.kaggle.com/code/shirshmall/cyberbullying-detection-text-classification-ml/notebook
#### Cyberbullying Detection - Notebook - Deep Learning Models: https://www.kaggle.com/code/shirshmall/text-multi-class-classification-e-commerce-data/notebook
#### Cyberbullying Detection - Notebook - EDA:  https://www.kaggle.com/code/shirshmall/eda-cyberbullying-detection/notebook
#### Product Segregation in E-commerce - Notebook: https://www.kaggle.com/code/shirshmall/text-multi-class-classification-e-commerce-data/notebook

#### Datasets:
1. E-Commerce Data: https://www.kaggle.com/datasets/venkatasubramanian/automatic-ticket-classification
2. Cyber Text Data: https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset

# Text Classification Project: Cyberbullying Detection and E-commerce Product Classification

## Introduction
This project ventures into the realm of Natural Language Processing (NLP) with a focus on text classification, addressing two distinct tasks: Cyberbullying Detection and E-commerce Product Classification. By harnessing a range of Machine Learning (ML), Deep Learning (DL), and Transformer models, the project aims to construct efficient text classification systems.

## Project Objectives
The primary objectives of this project are to:
1. Develop accurate text classification models for identifying cyberbullying content.
2. Create models for classifying E-commerce products based on their descriptions into multiple categories.

## Model Architectures and Text Embeddings
The project initiates with the establishment of model architectures and text embedding strategies. Two major approaches are pursued:
1. **ML Model-Based Approach**: This involves employing various tokenization and vectorization techniques such as Spacy Tokenizer, Word2Vec, and GloVe. Four ML models - Naive Bayes, Support Vector Machine (SVM), LightGMB, and XGBoost - are trained using different combinations of tokenization and vectorization methods.
2. **DL and Transformers Model-Based Approach**: Utilizing TensorFlow's TextVectorization, this approach involves DL architectures like Embedding + Flatten, Embedding + GlobalMaxPooling, Embedding + Bidirectional LSTM/GRU, Embedding + Conv1D + GlobalMaxPooling, Embedding + Bidirectional LSTM/GRU + Conv1D, and the Small BERT model.

## Data and Preprocessing
### Task 1: Cyberbullying Detection
For the binary classification task of cyberbullying detection, the dataset encompasses text samples from diverse social media platforms, each labeled as bullying or not. The data spans various forms of cyberbullying such as hate speech, aggression, insults, and toxicity. Data preprocessing steps entail the removal of URLs, HTML tags, punctuations, single characters, and numbers, as well as converting text to lowercase.

### Task 2: E-commerce Product Classification
The E-commerce classification dataset comprises product descriptions categorized into four classes: "Electronics," "Household," "Books," and "Clothing and Accessories." Preprocessing measures mirror those of Task 1, involving text cleaning and lowercasing for consistency.

## Model Training and Results
The training and results can be summarized as follows:

### Task 1: Cyberbullying Detection
1. **ML Model-Based Approach**:
   - **Spacy Tokenizer and Vectorizer**: F1-scores ranged from 0.849 to 0.856.
   - **Word2Vec and Gensim Vectorization**: Achieved F1-scores of LGBM: 0.8999, XGBoost: 0.922, Linear SVC: 0.930.
   - **GloVe Vectorization**: F1-scores spanned from 0.899 to 0.910.
   
2. **DL and Transformers Model-Based Approach**:
   - **Embedding + Bidirectional LSTM/GRU**: Yielded F1-scores over 0.9.
   - **Embedding + Conv1D + GlobalMaxPooling**: Similar F1-scores above 0.9.
   - **Small BERT Model**: Displayed competitive performance with F1-scores around 0.9.
   
### Task 2: E-commerce Product Classification
1. **ML Model-Based Approach**:
   - **Spacy Tokenizer and Vectorizer**: Achieved F1-scores between 0.620 and 0.710.
   - **Word2Vec and Gensim Vectorization**: F1-scores were LGBM: 0.900, XGBoost: 0.926, Linear SVC: 0.900.
   - **GloVe Vectorization**: Observed F1-scores around 0.690 to 0.700.
   
2. **DL and Transformers Model-Based Approach**:
   - **Embedding + Bidirectional LSTM/GRU**: Showcased F1-scores within the same range.
   - **Embedding + Conv1D + GlobalMaxPooling**: Demonstrated F1-scores close to 0.70.
   - **Small BERT Model**: Displayed competitive performance, achieving F1-scores near 0.70.

## Conclusion
The project's accomplishments underline the potential of text classification models in addressing real-world challenges like cyberbullying detection and product classification in E-commerce. Through careful model selection, embedding strategies, and preprocessing steps, the project achieves impressive results in both tasks. The combination of ML, DL, and Transformer methodologies underscores the versatility of text classification systems.


## Positive and Negative Outcomes
Positive outcomes of the project include:
- Successful implementation of various ML, DL, and Transformer-based models for text classification tasks.
- Achievement of high F1-scores, especially with Word2Vec, Gensim, and TF-IDF embeddings in both tasks.
- Effective handling of imbalanced data in Task 1, showcasing the models' adaptability.

Negative outcomes encompass:
- Poor performance of Naive Bayes in both tasks.
- Moderate performance of some models using Spacy and GloVe embeddings.
