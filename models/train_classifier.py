# import libraries
import sys
from sqlalchemy import create_engine
import sqlalchemy as db
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
import pickle
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])


def load_data(database_filepath):
    """Load and merge messages and categories datasets

    Args:
    database_filename: string. Filename for SQLite database

    Returns:
    X: Array. Array containing message input info.
    Y: Array. Array containing labels info.
    category_names: list of strings. List of category names.

    """
    engine = create_engine('sqlite:///' + database_filepath)
    df=pd.read_sql("messages", engine)
    X = df.message.values
    Y = df.iloc[:,4:-1].values
    category_names=df.columns[4:-1]
    return X,Y,category_names


def tokenize(text):

    """ Normalize, Tokenize and stem text dataset
    Args:
    text: String. Message Texts.

    Returns:
    stemmed: List of Strings.  List containing normalized and stemmed word tokens.


    """
    # Remove Punctuation
    text=re.sub(r"[^a-zA-Z0-9]"," ",text)

    # Tokenize words
    words=word_tokenize(text)

    # Remove Stop Words
    words=[w.lower().strip() for w in words if w not in stopwords.words('english')]

    # Stem Words
    stemmer = PorterStemmer()
    stemmed = [PorterStemmer().stem(w) for w in words]

    return stemmed

def performance_metric(y_test,y_pred):
    """Set up performance metrics including F1 score, precision and recall
    Args:
    y_test: Array. The array of actual labels.
    y_pred: Array. The array of predicted labels.

    Return:
    score: median of all multioutputclassification output.

    """
    score_list=[]
    for i in range(np.shape(y_test)[1]):
        f1=f1_score(y_true=list(y_test[:,i]),y_pred=list(y_pred[:,i]))
        score_list.append(f1)
    score=np.median(score_list)
    return score

def build_model():
    """ Build Machine Learning pipeline
    Args:
    None

    Returns:
    cv: Machine learning model. Leveraged gridsearch method to tune and
    uncover the optimal model.

    """
    # Build machine learning pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(LinearSVC()))])
    # Define the GridsearchCV parameters
    parameters = {
        'clf__estimator__class_weight':['balanced'],
        'clf__estimator__max_iter': [1000,1500,2000],
        'clf__estimator__C':[1,1.5,2]
    }
    #Create a scorer which is the evaluation of GridSearch
    f1_scorer=make_scorer(performance_metric)
    cv = GridSearchCV(pipeline, param_grid=parameters,scoring=f1_scorer,verbose=10)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """"Print the performance metrics
    Args:
    model: Machine learning model. Optimal model by Gridsearch.
    X_test: Array. Array includes input preprocessed messages.
    Y_test: Array. Array includes actual labels.
    category_names: list of strings. List of category names.

    Returns:
    None
    """
    # Predicted output based on test dataset
    Y_pred = model.predict(X_test)
    # Calculate performance metrics for each category
    metrics=[]
    for i in range(len(category_names)):
        precision=precision_score(list(Y_test[:,i]),list(Y_pred[:,i]))
        recall=recall_score(Y_test[:,i],Y_pred[:,i])
        accuracy=accuracy_score(Y_test[:,i],Y_pred[:,i])
        f1=f1_score(Y_test[:,i],Y_pred[:,i])

        metrics.append([precision,recall,accuracy,f1])

    # Metrics output
    metrics_output=pd.DataFrame(data=metrics,columns=['Precision','Recall','Accuracy','f1 score'],index = category_names)
    print (metrics_output)


def save_model(model, model_filepath):
    """ Saved optimal model
    Args:
    model: Machine learning model. Optimized model output by Gridsearch.
    model_filepath: string. The path that model will be saved.

    Returns:
    None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
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
