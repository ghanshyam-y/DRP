import pandas as pd
import sys
from sqlalchemy import create_engine


def load_data(categories_path, messages_path):
    categories = pd.read_csv(categories_path)
    messages = pd.read_csv(messages_path)
    return categories, messages

def merge_data(categories, messages):
    merged_data = messages.merge(categories, how='outer', on='id')
    return merged_data

def clean_data(data):
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
    return data


def save_data(data):
    engine = create_engine('sqlite:///mydata.db')
    data.to_sql('mydata', engine, index=False)

def run_pipeline(categories_path, messages_path, output_path):
    categories, messages = load_data(categories_path, messages_path)
    data = merge_data(categories, messages)
    data = clean_data(data)
    save_data(data)

if __name__ == '__main__':
    categories_path, messages_path = sys.argv[1:]
    run_pipeline(categories_path, messages_path)