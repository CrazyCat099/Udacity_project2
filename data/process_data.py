# import libraries 
import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    load and merge the messages_df and categories_df
    Returns:
    df: An unified dataframe
    
    """
    # load data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets on common id
    df = messages.merge(categories, how ='outer', on =['id'])
    return df

def clean_data(df):
    """
    Cleans the dataframe.

    """
    # create a dataframe of the individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select first row of the categories dataframe
    row_1st = categories.head(1)
    # since the value of columns has 1 character 0 or 1, we use the last character as the value and convert to int, and the first n-2 character as name columns
    category_colnames = row_1st.applymap(lambda x: x[:-2]).iloc[0,:]
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop related == 2 because it has less<1% value
    df= df[df['related']!=2]
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df
    
def save_data(df, database_filepath):
    """Stores df in a SQLite database."""
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')  


def main():
    """Loads data, cleans data, saves data to database"""
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
