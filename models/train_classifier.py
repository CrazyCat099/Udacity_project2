import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    """
    Loads data from SQLite database.
    Parameters:
    database_filepath: Filepath to the database
    Returns: X: feature ; Y: target; category name
    """
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("disaster_messages", con=engine)
    #X = df[['message','genre']]
    X= df['message']
    Y = df.iloc[:, 4:]  
    category_name = list(Y.columns)
#     print(category_name)
#     print(df.columns)
    return X,Y,category_name


def tokenize(text):
    """
    tokenizes and lemmatizes text.
    input: text
    return: cleaned token
    """
    token= word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    
    clean_token =[]
    
    for tk in token:
        
        clean_tk=  lemmatizer.lemmatize(tk).lower().strip()
        clean_token.append(clean_tk)
    return clean_token

def build_model():
    pipeline= Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
         ])
    
    grid_pipe_parameters= {
    #'clf__estimator__max_features': ['auto', 'sqrt'],
    #'clf__estimator__max_depth' : [10],
    'clf__estimator__n_estimators': [50,100]
    }
    
    cv = GridSearchCV(pipeline, param_grid=grid_pipe_parameters, verbose=2)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Returns:
    Classification report for each column
    """

    y_pred = model.predict(X_test)
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))

    
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()