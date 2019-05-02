# Disaster-responses-pipeline

Implemented a machine learning pipeline to categorize emergency messages based on the needs communicated by the sender.

## Installation

There is no necessary libraries to run the code, except the ones includes in each file. The code runs with no issues 
using Python 3.6 or newer versions.

## Project Motivation

This project was part of the Data Science Udacity Program and needed to be completed in order to obtain a certificate.

## File Description

There are 4 directories that are part of this project. Details below:

**Data**:
* categories.csv, messages.csv: Contains the message data to process.
* process_data.py: Contains the ETL process of the csv files below. Reads the datasets, clean the data and then stores 
  it in a SQLite database. Usage:
  
  ``python process_data.py messages.csv categories.csv [database_name]``
  
  Example: ``python process_data.py messages.csv categories.csv DisasterResponse.db``
  
**app**:
* run.py: Flask file that runs web app
* template: 
    * master.html: Main page of the web app.
    * go.html: Classification result page of web app.
    
**models**:
* train_classifier.py: Contains the machine learning pipeline, which uses NLTK, scikit-learn's Pipeline and GridSearchCV 
  to output a final model that uses the message column to predict classifications for 36 categories (multi-output 
  classification). Then, exports the resulting model to a pickle file. 
  Usage:
  
  ``python train_classifier.py [database_name] [model_export_file]``
  
  Example: ``python train_classifier.py DisasterResponse.db classifier.pkl``
    
**notebooks**:
* ETL Pipeline Preparation.ipynb: ETL process of the data. The process_data.py file is based on this file. 
* ML Pipeline Preparation.ipynb: Contains steps needed for the ML pipeline. The train_classifier.py file is based on
  this file.  
  
## Results

The main findings are found in the ETL Pipeline Preparation.ipynb and ML Pipeline Preparation.ipynb where markdown cells
where used to walk through all the steps. 