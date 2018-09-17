import pandas as pd
import re
import sys
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def load_data(file_path):
    """
    Loads data from the given sqlite data path
    :param file_path: String
    :return data: DataFrame
    """
    print('Loading data...\n    PATH: {}\n'.format(file_path))
    engine = create_engine('sqlite:///{}'.format(file_path))
    df = pd.read_sql_table('mydata1', engine)
    X = df['message'].values
    y = df.drop(['id','message','original','genre'], axis=1).values
    categories = df.columns[4:]
    return X, y, list(categories)

def tokenize(text):
    """
        Tokenizes the text provided
        :param text: String
        :return: List
    """
    text = re.sub(r"[^A-Za-z]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model(tokenizer):
    """
    Builds a model pipeline
    :param tokenizer: function
    :return: model
    """
    print('Building Model')
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenizer)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    # Parameters to grid search
    parameters = {'tfidf__use_idf': (True, False),
                  'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__smooth_idf': (True, False)}

    # Grid search
    model_pipeline = GridSearchCV(pipeline, parameters)
    return model_pipeline

def train(X, y, categories, model):
    """
    Trains a given model
    :param X: Data
    :param y: labels
    :param categories: label_names
    :param model: model
    :return: model
    """
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Print results
    for index, category in enumerate(categories):
        print(classification_report(y_test[:, index], y_pred[:, index], target_names=[category]))
    return model

def export_model(model, model_filepath):
    """
    Save the model to a pickle file
    :param model: model
    :param model_filepath: String
    :return:
    """
    # Save the model
    pickle.dump(model, open(model_filepath, "wb"))


def run_pipeline(data_file, model_filepath):
    """
    Runs the ML pipeline
    :param data_file: String
    :param model_filepath: String
    :return:
    """
    X, y, categories = load_data(data_file)  # run ETL pipeline
    model = build_model(tokenizer=tokenize)  # build model pipeline
    model = train(X, y, categories, model)  # train model pipeline
    export_model(model, model_filepath)  # save model

if __name__ == '__main__':

    if len(sys.argv) == 3:
        data_file, model_filepath = sys.argv[1:]
        run_pipeline(data_file=data_file, model_filepath=model_filepath)

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


