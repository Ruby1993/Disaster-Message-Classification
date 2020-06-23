#import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np


def load_data(messages_filepath, categories_filepath):
    """ Load and merge messages and category dataset
    Args:
    messages_filepath: String. Path of messages dataset
    categories_filepath: String. Path of categories dataset

    Return:
    df: Dataframe. The output includes messages and category datasets
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df=messages.merge(categories,on=['id'],how='left')
    return df


def clean_data(df):
    """ Clean Dataset by processing categories dataset and removing duplicates
    Args:
    df: Dataframe. Dataframe includes messages and category datasets.

    Return:
    df: DataFrame.Dataframe includes cleaned messages and category datasets.

    """
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x:x[:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Remove the rows with value 2 or the columns with only one value
    #categories=categories[categories['related']!=2]
    for i in category_colnames:
        if len(np.unique(categories[i]))==1:
            categories.drop(i,axis=1,inplace=True)
        elif len(np.unique(categories[i]))>2:
            categories=categories[categories[i].isin([0,1])]
        else:
            categories=categories
    # drop the original categories column from `df`
    df=df.drop(['categories'],axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1,join='inner')
    # check number of duplicates
    df["is_duplicate"]=df.duplicated(keep='first')
    # drop duplicates
    df=df[df["is_duplicate"]==False]
    df=df.drop(['is_duplicate'],axis=1)

    return df


def save_data(df, database_filename):
    """saved processed data into a SQLite Database
    Args:
    df: Dataframe. The dataframe includes processed dataset.
    database_filename: String. Name for the output database.

    Returns:
    None
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False,if_exists='replace')


def main():
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
