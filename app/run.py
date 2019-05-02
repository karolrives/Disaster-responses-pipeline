import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import re


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(re.sub(r"[^a-zA-Z0-9]", " ", tok.lower()), pos='v').strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../Data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/final_model_v3.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    df_group = df.drop(columns=['id','message','original']).groupby('genre').sum()
    df_group = df_group.T
    df_group['total'] = df_group.direct + df_group.news + df_group.social
    df_group.sort_values(by='total', ascending=False, inplace=True)
    df_group.drop(columns=['total'], inplace=True)



    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x = df_group.index.tolist(),
                    y = df_group['news'],
                    name = 'news',
                    width = 0.8,
                    #orientation = 'h'

                ),
                Bar(
                    x = df_group.index.tolist(),
                    y = df_group['direct'],
                    name = 'direct',
                    #orientation='h',
                    width=0.8,
                ),
                Bar(
                    x = df_group.index.tolist(),
                    y = df_group['social'],
                    name='social',
                    #orientation='h',
                    width=0.8,
                )
            ],

            'layout': {
                'title': 'Distribution of messages by category',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title': ' '
                },
                'barmode': 'stack',
                'legend': {
                    'orientation': 'h',
                    'x': 0.33,
                    'y': 1.1
                },
                'margin': {
                    #'l': 1
                },
                'width': 740
            }

        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()