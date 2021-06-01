
import math
import os

import numpy as np
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def pre_process_data(data, null_threshold):
    """
    Drops Date and Unix Date columns from the data.
    Drops the columns which has null values more than specified null_threshold.
    Replaces infinite values with NAN.
    Drops the rows which has null values.

    Parameters
    ----------
    data : dataframe

    null_threshold : numeric
        numeric value describing the amount of null values that can be present.

    Returns
    -------
    data : dataframe
        an updated dataframe after performing all the opertaions.
    """

    data.drop(columns=['Unix Date', 'Date'], axis=1, inplace=True)
    total = data.shape[0]
    for col in data.columns:
        if null_threshold * total / 100 < data[col].isnull().sum():
            data.drop(columns=[col], axis=1, inplace=True)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(axis=0, inplace=True)
    return data


def dependent_column(data, column):
    """
    Removes all the Next Day columns.
    Removes all the non Growth Rate Columns (GR)
    add the predictor column to list of columns.

    Parameters
    ----------
    data : dataframe

    column : string
        name of the predictor column 

    Returns
    -------
    data : dataframe
        an updated dataframe after performing all the opertaions.
    column : string
        name of the predictor column
    """
    cols = [col for col in data.columns if "next" not in col.lower()
            and col.lower().endswith("gr")]
    cols.append(column)
    data = data[cols]
    return (data, column)


def error_metrics(y_true, y_pred):
    rmse = math.sqrt(metrics.mean_squared_error(y_true, y_pred))
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    return {"root_mean_squared_error": rmse, "mean_absolute_error": mae, "mean_squared_error": mse}


def create_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score, "confusion matrix": cm}


def LogisticModel(df, column, rate, C, penalty):
    df["Target"] = df["Next Day Close Price GR"].apply(
        lambda x: 1 if x >= rate else 0)
    X = df.drop(columns=["Target", column])
    Y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=0)
    logmodel = LogisticRegression(penalty=penalty, C=C, random_state=0)
    logmodel.fit(X_train, y_train)
    y_pred = logmodel.predict(X_test)

    result = {}
    error = error_metrics(y_test, y_pred)
    confusion = create_confusion_matrix(y_test, y_pred)
    result.update(error)
    result.update(confusion)
    return result


def KNNClassification(df, column, rate, metric, n_neighbors, weights, algorithm):
    df["Target"] = df[column].apply(lambda x: 1 if x >= rate else 0)
    X = df.drop(columns=["Target", column])
    Y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=0)
    knn = KNeighborsClassifier(
        metric=metric, n_neighbors=n_neighbors, weights=weights)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    result = {}
    confusion = create_confusion_matrix(y_test, y_pred)
    error = error_metrics(y_test, y_pred)
    result.update(error)
    result.update(confusion)
    return result


def BPNNClassification(df, column, rate, epochs, batch_size):
    def create_model_b(input_dim, layers=3, units=32):
        model = Sequential()
        model.add(Dense(units=units, input_dim=(input_dim), activation='relu'))
        model.add(Dense(units=units, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(loss="binary_crossentropy", optimizer="Adam",
                      metrics=tf.keras.metrics.Precision())
        return model
    df["Target"] = df[column].apply(lambda x: 1 if x >= rate else 0)
    X = df.drop(columns=["Target", column])
    Y = df["Target"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=0)
    input_dim = x_train.shape[1]

    x_train = np.array(x_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))
    y_test = np.array(y_test)
    y_test = np.reshape(y_test, (y_test.shape[0], 1))

#     model = KerasClassifier(build_fn = create_model_b, batch_size=batch_size, epochs=epochs,input_dim = input_dim)
    model = create_model_b(input_dim)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(x_test, y_test), shuffle=False, verbose=0)
    result = {}

    y_pred = model.predict(x_test)
    y_pred = np.array(y_pred)

    result.update({'actual': y_test, 'pred': y_pred})
    y_pred = np.reshape(y_pred, (y_pred.shape[0], 1)).round()

    error = error_metrics(y_test, y_pred)
    confusion = create_confusion_matrix(y_test, y_pred)
    result.update(error)
    result.update(confusion)
    return result


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def CNNClassification(df, column, rate, optimizer, activation, dropout_rate, neurons, layers, batch_size, epochs):

    def create_model(input_shape, optimizer='adam', activation='relu', dropout_rate=0.2, neurons=32, layers=3):
        model = Sequential()
        model.add(Conv1D(neurons, 1, activation=activation,
                  input_shape=input_shape))
        model.add(Flatten())
        for i in range(layers):
            model.add(Dense(neurons, activation=activation))
            model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation=activation))
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=tf.keras.metrics.Precision())
        return model

    def reshape_data(x_train, x_test, y_train, y_test):
        x_train = np.array(x_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        y_test = np.array(y_test)
        y_test = np.reshape(y_test, (y_test.shape[0], 1))

        return (x_train, x_test, y_train, y_test)

    df["Target"] = df[column].apply(lambda x: 1 if x >= rate else 0)
    X = df.drop(columns=["Target", column])
    Y = df["Target"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=0)
    (x_train, x_test, y_train, y_test) = reshape_data(
        x_train, x_test, y_train, y_test)
    input_shape = (x_train.shape[1], 1)
    model = create_model(input_shape, optimizer=optimizer, activation=activation, dropout_rate=float(
        dropout_rate), neurons=int(neurons), layers=int(layers))
    history = model.fit(x_train, y_train, epochs=int(epochs), batch_size=int(
        batch_size), validation_data=(x_test, y_test), shuffle=False, verbose=0)
    y_pred = model.predict(x_test)
    s = 0.01
    t = 0.5
    while True:
        pred = np.array(y_pred > t, int)
        res = confusion_matrix(y_test, pred)
        if res[1][1] >= 100 or t <= 0:
            break
        t = t - s
    result = {}
    result.update({"threshold": t})

    confusion = create_confusion_matrix(y_test, pred)
    result.update(confusion)
    result.update({'epochs': epochs, 'batch_size': batch_size})
    result.update({'optimizer': optimizer, 'activation': activation,
                   'dropout_rate': dropout_rate, 'neurons': neurons, 'layers': layers})
    return result


def RNNClasification(df, column, rate, optimizer, activation, dropout_rate, neurons, layers, batch_size, epochs):
    def build_lstm(input_shape, optimizer, activation, dropout_rate, neurons, layers):
        model = Sequential()
        model.add(LSTM(neurons, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(neurons))
        model.add(Dropout(dropout_rate))
        for _ in range(layers):
            model.add(Dense(neurons))
            model.add(Dropout(dropout_rate))
        model.add(Dense(units=1, activation=activation))
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer, metrics=[tf.metrics.Precision()])
        return model

    def reshape_data(x_train, x_test, y_train, y_test, units=30):
        my_x_train = list()
        my_y_train = list()
        my_x_test = list()
        my_y_test = list()
        for i in range(x_train.shape[0]-units):
            my_x_train.append(x_train.iloc[i:i+units, :])
            my_y_train.append(y_train.iloc[i+units, ])

        my_x_train = np.array(my_x_train)
        my_x_train = np.reshape(
            my_x_train, (my_x_train.shape[0], my_x_train.shape[1], my_x_train.shape[2]))

        my_y_train = np.array(my_y_train)
        my_y_train = np.reshape(my_y_train, (my_y_train.shape[0], 1))

        for i in range(x_test.shape[0]-units):
            my_x_test.append(x_test.iloc[i:i+units, :])
            my_y_test.append(y_test.iloc[i+units, ])

        my_x_test = np.array(my_x_test)
        my_x_test = np.reshape(
            my_x_test, (my_x_test.shape[0], my_x_test.shape[1], my_x_test.shape[2]))

        my_y_test = np.array(my_y_test)
        my_y_test = np.reshape(my_y_test, (my_y_test.shape[0], 1))

        return (my_x_train, my_x_test, my_y_train, my_y_test)

    def split_dataset(X, Y, t):
        tr = int(len(X)*t)
        tt = len(X) - tr
        xtr = X[:tr]
        xtt = X[tr:tr+tt]
        ytr = Y[:tr]
        ytt = Y[tr:tr+tt]
        return (xtr, xtt, ytr, ytt)

    df["Target"] = df[column].apply(lambda x: 1 if x >= rate else 0)
    X = df.drop(columns=["Target", column])
    Y = df["Target"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=0)
    (x_train, x_test, y_train, y_test) = reshape_data(
        x_train, x_test, y_train, y_test)
    input_shape = (x_train.shape[1], x_train.shape[2])
    model = KerasClassifier(build_lstm, input_shape=input_shape, optimizer=optimizer,
                            activation=activation, dropout_rate=dropout_rate, neurons=neurons, layers=layers)
    history = model.fit(x_train, y_train, epochs=int(epochs), batch_size=int(
        batch_size), validation_data=(x_test, y_test), shuffle=False, verbose=0)
    y_pred = model.predict(x_test)
    result = {}
    confusion = create_confusion_matrix(y_test, y_pred)
    result.update(confusion)
    result.update({'epochs': epochs, 'batch_size': batch_size})
    result.update({'optimizer': optimizer, 'activation': activation,
                  'dropout_rate': dropout_rate, 'neurons': neurons, 'layers': layers})
    return result


def SVMClassification(df, column, rate, C, gamma, kernel):
    df["Target"] = df[column].apply(lambda x: 1 if x >= rate else 0)
    X = df.drop(columns=["Target", column])
    Y = df["Target"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=0)
    model = SVC(kernel=kernel, gamma=gamma, C=C)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    result = {}
    error = error_metrics(y_test, y_pred)
    confusion = create_confusion_matrix(y_test, y_pred)
    result.update(error)
    result.update(confusion)
    return result
