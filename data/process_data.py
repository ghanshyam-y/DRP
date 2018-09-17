import pandas as pd
import sys
from sqlalchemy import create_engine


def load_data(categories_path, messages_path):
    """
    Loads the categories and messages data
    :param categories_path: String
    :param messages_path: String
    :return: DataFrame, DataFrame
    """
    categories = pd.read_csv(categories_path)
    messages = pd.read_csv(messages_path)
    return categories, messages

def merge_data(categories, messages):
    """
    Merges the categories and messages data into one dataFrame
    :param categories: DataFrame
    :param messages: DataFrame
    :return: DataFrame
    """
    merged_data = messages.merge(categories, how='outer', on='id')
    return merged_data

def clean_data(data):
    """
    Cleans the data:
    :param data: DataFrame
    :return: DataFrame
    """
    categories = data['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # get categories
    category_colnames = row.apply(lambda value: value[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.extract('[a-z_-]*(\d)')
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    data.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    data = pd.concat([data, categories], axis=1)
    # drop duplicates
    data.drop_duplicates(inplace=True)
    data.fillna(0)
    return data


def save_data(data, database_filepath):
    """
    Save data to a sqlite database
    :param data: DataFrame
    :param database_filepath: String
    :return:
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    data.to_sql('mydata1', engine, index=False)

def run_pipeline(categories_path, messages_path, database_filepath):
    """
    Runs the ETL pipeline
    :param categories_path:
    :param messages_path:
    :param database_filepath:
    :return:
    """
    categories, messages = load_data(categories_path, messages_path)
    data = merge_data(categories, messages)
    data = clean_data(data)
    save_data(data, database_filepath)

if __name__ == '__main__':

    if len(sys.argv) == 4:
        categories_path, messages_path, database_filepath = sys.argv[1:]
        run_pipeline(categories_path, messages_path, database_filepath)

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')