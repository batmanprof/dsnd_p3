# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """Loading data"""
    # load data from database
    if database_filepath[:10] != 'sqlite:///':
        database_filepath = 'sqlite:///'+database_filepath    
    engine = create_engine(database_filepath)
    df = pd.read_sql_table("Data", con=engine)
    X = df['message']
    y = df[df.columns[4:]].values
    cols = df.columns[4:]
    return X,y,cols


def tokenize(text):
    """Preparing the textual data 
    * lower casing
    * tokenizing
    * deleting stop words
    * lemmatizing
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())    
    # Tokenizing text
    words = word_tokenize(text)
    # Removing stop words
    words = [w for w in words if w not in stopwords.words("english")]   
    # Lemmatizing
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]    
    return words

def build_model():
    """Building model and setting parameters"""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),    
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters =  { 'tfidf__use_idf': (True, False) }
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2)    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluating a trained model"""
    Y_pred = model.predict(X_test)
    for i in range(36):
        print(category_names[i], ':\n', classification_report(Y_test[i], Y_pred[i]))


def save_model(model, model_filepath):    
    """Saving model"""
    joblib.dump(model, model_filepath, compress = 1)

def main():
    """Main function"""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
        
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