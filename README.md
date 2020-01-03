# Disaster Response Pipeline Project

## Summary
This project is made for the Data Scientist Nano Degree. The puprose of this project is
classifying messages and find those which indicate that some help needed.
Main points:
* Preprocess the provided 
* Create and train a model on the provided data
* Deploy the model in a web-app

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/clf.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



## File Structure 

```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- clf.pkl  # saved model

- notebooks
|- ETL Pipeline Preparation.ipynb Preparation notebook for process_data.py
|- ML Pipeline Preparation.ipynb Preparation notebook for train_classifier.py

- README.md (You are reading this now.)
```