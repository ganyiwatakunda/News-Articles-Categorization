import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost
from sklearn.metrics import classification_report
from sklearn import metrics
import time
from nltk.tokenize import sent_tokenize


import nltk
nltk.download('punkt')

# Read the dataset from csv file
df1 = pd.read_csv('BBC News Train.csv')
df1.head()

# Find all category
category = list(df1['Category'].unique())
category

# Text preprocessing
def preprocess(text):
    
    """
    Function: split text into words and return the root form of the words
    Args:
      text(str): the article
    Return:
      lem(list of str): a list of the root form of the article words
    """
        
    # Normalize text
    text = re.sub(r"[^a-zA-Z]", " ", str(text).lower())
    
    # Tokenize text
    token = word_tokenize(text)
    
    # Remove stop words
    stop = stopwords.words("english")
    words = [t for t in token if t not in stop]
    
    # Lemmatization
    lem = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return lem

# Find the common words in each category
def find_common_words(df, category):
        
    """
    Function: find the most frequent words in the category and return the them
    Args:
      df(dataframe): the dataframe of articles
      category(str): the category name
    Return:
      the most frequant words in the category
    """
        
    # Create dataframes for the category
    cat_df = df[df["Category"]==category]
    
    # Initialize words list for the category
    words = [word for tokens in cat_df["Preprocessed_Text"] for word in tokens]
    
    # Count words frequency
    words_counter = Counter(words)
 
    return words_counter.most_common(10)


# Train and evaluate model

    
    


def main():
    st.title("News Articles Categorization")
    st.header("Overview")
    st.write("In this project, I will apply machine learning algorithms to news collected from CNN News website to construct a model that classifies the news into groups.")

    # Load the dataset
    df = pd.read_csv("BBC News Train.csv")

   # Display the articles by category
    st.subheader("Articles by Category")
    selected_category = st.selectbox("Select a category", df["Category"].unique())
    selected_articles = df[df["Category"] == selected_category][["Text", "ArticleId"]]

    

    for index, row in selected_articles.iterrows():
              if row['Category'] in ['business', 'politics', 'entertainment', 'sports']:
                    sentences = word_tokenize(row['Text'])
                    truncated_text = ' '.join(sentences[:]10)  # Display the first 2 sentences
                    st.write(f"{truncated_text}... [Read More]({row['ArticleId']})")


if __name__ == '__main__':
    main()
