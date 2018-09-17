# Disaster Response Pipeline

This project builds an API that classifier disaster messages so that the appropriate disaster relief agency can help efficiently. 

A data set containing real messages sent during disaster events are used to create a machine learning pipeline to categorize these events and subsequently inform the respective relief agency.

There are three major component of this project:
 
 1) ETL Pipeline
 2) Machine Learning Pipeline
 3) Flask App
 
 <h4>ETL Pipeline</h4>
 The first part of the data pipeline is the Extract, Transform, and Load process. A disaster dataset, containing messages and labeled categories, is loaded, cleaned and stored in a SQLite database.
 
 The ETL pipeline can be run from the data directory:
 ```$xslt
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

```
 
 <h4> Machine Learning Pipeline </>
 
The cleaned data stored in the SQLite databas is used to train a model. A machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV is used to output a final model. The model is then exported as a pickle file.
The ML pipeline can be run from the models directory:
```$xslt
python train_classifier.py ../data/DisasterResponse.db classifier.pkl

```

<h4> Flask App </h4>

A flask app is provided that can be used to classify any message into one of the 36 categories of events. Data visualizations are also provided for data analysis.
The flask app can be run from the app directory:
```$xslt
python run.py

```