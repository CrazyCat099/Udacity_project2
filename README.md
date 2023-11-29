# Udacity_project2

In this Project, I aim to classify a diasater message into classes. I created a ML pipeline that take a message as input and classify into several kind of categories and a demo app to visualize the result of model. This Project is also belong to Udacity Datascience nanodegree.


## File Descriptions
###app

template
* master.html # main page of web app
* run.py #  file runs app

### data

- disaster_categories.csv # Target data
 - disaster_messages.csv # text data 
- process_data.py # data cleaning
 - DisasterResponse.db # database

### models

- train_classifier.py # ML pipeline
- classifier.pkl # saved model

### README.md

## How to run 
### To create a processed sqlite db
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
### To train and save a pkl model
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
### To deploy the application locally
python run.py
