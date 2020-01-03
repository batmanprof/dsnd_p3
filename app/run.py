import json
import plotly
import pandas as pd
import nltk

import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from collections import Counter 

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())    
    tokens = word_tokenize(text)
    # Removing stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]       
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Data', engine)

# load model
model = joblib.load("../models/clf.pkl")

# preparation for graphs
print('Preparation for graphs...')
lengths=[]
for s in df['message']:
    lengths.append(len(s))
print('Done.')
lengths.sort(reverse=True)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
                    
    dis_type = ['floods','storm','fire','earthquake']
    nr_of_dis = [df['floods'].sum(),df['storm'].sum(),df['fire'].sum(),df['earthquake'].sum()]
    
    nr_of_words = [lengths[i] for i in range(5)]
    place = [1,2,3,4,5]
    
    
    # create visuals
    # Below is an example - modify to create your own visuals
    graphs = [
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
        },
        {
            'data': [
                Bar(
                    x=dis_type,
                    y=nr_of_dis
                )
            ],

            'layout': {
                'title': 'Number of messages about different types of disasters',
                'yaxis': {
                    'title': "Number"
                },
                'xaxis': {
                    'title': "Type of disaster"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=place,
                    y=nr_of_words
                )
            ],

            'layout': {
                'title': 'Top-5 longest messages',
                'yaxis': {
                    'title': "Number of characters"
                },
                'xaxis': {
                    'title': "Place"
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