Sentiment Analysis Project
==========================

This project is a sentiment analysis classifier built using Python, based on text data from tweets. It follows the steps from an article titled ["Getting Started with Sentiment Analysis: A Step-by-Step Guide"](https://medium.com/@swayampatil7918/getting-started-with-sentiment-analysis-a-step-by-step-guide-1a16085688a7) and uses several machine learning techniques, including TF-IDF vectorization and a Support Vector Classifier (SVC) to predict the sentiment of a given text. I included a lot of comments in the main file for learning purposes.

Project Overview
----------------

The goal of this project is to classify the sentiment of tweets into three categories: **positive**, **negative**, or **neutral**. The machine learning model is trained using a dataset of tweets, with the sentiment provided as labels. It includes:

*   Preprocessing the text (lowercasing, tokenization, stopword removal)
    
*   Vectorizing the text data using **TF-IDF**
    
*   Training a Support Vector Machine (**SVM**) for classification
    
*   Evaluating the model using metrics like precision, recall, F1-score, and accuracy.
    

Technologies Used
-----------------

*   **Python 3.11**
    
*   **Pandas**: For data manipulation and analysis.
    
*   **NLTK**: For text preprocessing (tokenization, stopwords).
    
*   **Scikit-learn**: For machine learning algorithms and evaluation metrics.
    
*   **Jupyter Notebook/Terminal**: For development and testing.

Dataset
-------

The dataset used in this project is a CSV file named Tweets.csv, which contains a collection of tweets labeled with sentiment as **positive**, **negative**, or **neutral**. [It is from Mubasher Bajwa on Kaggle.](https://www.kaggle.com/code/mubasherbajwa/complete-guide-to-twitter-sentiment-analysis-nlp)

### Columns:

*   **text**: The tweet text.
    
*   **sentiment**: The sentiment label (positive, negative, neutral).
    

Model Training Process
----------------------

The project follows these key steps:

1.  **Data Preprocessing**:
    
    *   Text is converted to lowercase.
        
    *   Tokenization is applied using NLTK.
        
    *   Stopwords are removed to focus on meaningful words.
        
2.  **Feature Extraction**:
    
    *   The TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer is used to convert the text data into numerical form, capturing the importance of each word in relation to the corpus.
        
3.  **Model Training**:
    
    *   An **SVM classifier** (Support Vector Machine) is used to train on the TF-IDF vectors and corresponding sentiment labels.
        
4.  **Evaluation**:
    
    *   The model is evaluated on test data using metrics like precision, recall, F1-score, and accuracy.
        

Results
-------

The sentiment analysis model achieved an accuracy of **~70%** on the test set. Below is the classification report showing the model's performance across different sentiment categories:

Acknowledgments
---------------

This project was developed by following a tutorial from an article. Special thanks to the author of the tutorial for providing a clear and comprehensive guide to building a sentiment analysis model. The tutorial helped in learning the application of text preprocessing, TF-IDF, and SVM for classification.
