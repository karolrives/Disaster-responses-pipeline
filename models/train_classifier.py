import sys
import pandas as pd
import numpy as np
import sqlalchemy
import re
from time import time

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import classification_report

import pickle

def load_data(database_filepath):
    database_filepath = 'sqlite:///{}'.format(database_filepath)
    engine = sqlalchemy.create_engine(database_filepath)

    df = pd.read_sql_table('messages', con=engine)
    X = df['message'].values
    y = df.drop(columns=['id', 'message', 'original', 'genre']).values
    labels = df.drop(columns=['id', 'message', 'original', 'genre']).columns

    return X, y, labels

def tokenize(text):
    # tokenize text
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(re.sub(r"[^a-zA-Z0-9]", " ", tok.lower())).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])

    parameters = {
        'vect__max_features': (None, 5000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10, 20, 50],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, n_jobs=-1, verbose=2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    y_pred = model.predict(X_test)

    print("\nBest Parameters: \n", model.best_params_)

    print("\n",classification_report(Y_test, y_pred, target_names=category_names))



def save_model(model, model_filepath):

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
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