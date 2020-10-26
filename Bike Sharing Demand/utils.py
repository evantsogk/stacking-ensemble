import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor
from yellowbrick.model_selection import FeatureImportances

"""This module provides various functions that facilitate the handling of data and evaluation of predictions.
"""
_author_ = "Evangelos Tsogkas p3150185"


def load_data(filename_train, filename_test):
    """Loads the train and test set, plots the feature importances extracted from Extra Trees and creates the final
    training set by removing unnecessary features. It returns separate target value sets for the 'casual' and
    'registered' values, instead of a single one for the 'count' values.
    """
    df_train = pd.read_csv(filename_train)
    df_test = pd.read_csv(filename_test)

    # change column names
    df_train.rename(columns={'weathersit': 'weather', 'mnth': 'month', 'hr': 'hour', 'yr': 'year', 'hum': 'humidity',
                             'cnt': 'count'}, inplace=True)
    df_test.rename(columns={'weathersit': 'weather', 'mnth': 'month', 'hr': 'hour', 'yr': 'year', 'hum': 'humidity',
                            'cnt': 'count'}, inplace=True)

    # feature importances
    plt.figure()
    model = ExtraTreesRegressor(n_jobs=-1, random_state=0, n_estimators=500, max_depth=26)
    viz = FeatureImportances(model)
    viz.fit(df_train.drop(['casual', 'registered', 'count'], axis=1), df_train['count'])

    # create train-test set
    y_casual = df_train['casual']
    y_registered = df_train['registered']
    x_train = df_train.drop(['casual', 'registered', 'count', 'atemp', 'windspeed'], axis=1)
    x_test = df_test.drop(['atemp', 'windspeed'], axis=1)

    return x_train, y_casual, y_registered, x_test


def one_hot_encode(x_train, x_test):
    """Returns the one hot encoded version of the categorical data in the training set.
    """
    # categories per feature
    categories = []
    for i in ['season', 'month', 'hour', 'weekday', 'weather']:
        categories.append(np.unique(x_train[i]))

    enc = OneHotEncoder(categories=categories, sparse=False)

    # one hot encode the categorical columns
    categorical_x = x_train[['season', 'month', 'hour', 'weekday', 'weather']].values
    categorical_df_test = x_test[['season', 'month', 'hour', 'weekday', 'weather']].values
    x_train_enc = enc.fit_transform(categorical_x)
    x_test_enc = enc.fit_transform(categorical_df_test)

    # add the rest of the columns
    x_train_enc = np.concatenate([x_train_enc, x_train[['year', 'holiday', 'workingday', 'temp', 'humidity']].values],
                                 axis=1)
    x_test_enc = np.concatenate([x_test_enc, x_test[['year', 'holiday', 'workingday', 'temp', 'humidity']].values],
                                axis=1)

    return x_train_enc, x_test_enc


def convert_negative_to_zero(y_pred):
    """Converts negative predictions to zero.
    """
    for i, y in enumerate(y_pred):
        if y_pred[i] < 0:
            y_pred[i] = 0


def rmsle_score(y_true, y_pred):
    """Returns the root mean squared logarithmic error.
    """
    convert_negative_to_zero(y_pred)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def export_submission(y_pred, filename):
    """Exports the submission csv with the predictions of the test set.
    """
    convert_negative_to_zero(y_pred)
    submission = pd.DataFrame()
    submission["Id"] = range(y_pred.shape[0])
    submission["Predicted"] = y_pred
    submission.to_csv(filename, index=False)


def regression_plot(y_true, y_pred):
    """Creates a regression plot for the actual and predicted values.
    """
    plt.figure()
    ax = sn.regplot(y_true, y_pred, x_bins=200)
    ax.set(title="Comparison between the actual vs predicted values")


def plot_error_distribution(y_true, y_pred):
    """Plots the error distribution of the predictions.
    """
    plt.figure()
    error = np.subtract(y_true, y_pred)
    plt.hist(error, bins=200)
    plt.title("Prediction Error Distribution")
