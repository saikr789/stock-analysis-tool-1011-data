import ray
import pandas as pd
import numpy as np
import os
import re
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn import metrics
import traceback
import warnings
warnings.filterwarnings("ignore")


def pre_process_data(df, null_threshold):
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

    df.drop(columns=['Date'], axis=1, inplace=True)
    total = df.shape[0]
    for col in df.columns:
        if null_threshold * total / 100 < df[col].isnull().sum():
            df.drop(columns=[col], axis=1, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, inplace=True)
    return df


def error_metrics(y_true, y_pred):
    rmse = metrics.mean_squared_error(y_true, y_pred) ** 0.5
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2_score = metrics.r2_score(y_true, y_pred)
    return {"root_mean_squared_error": rmse, "mean_absolute_error": mae, "mean_squared_error": mse, "r2_score": r2_score}


def split_dataset(X, Y, t):
    tr = int(len(X)*t)
    tt = len(X) - tr
    xtr = X[:tr]
    xtt = X[tr:tr+tt]
    ytr = Y[:tr]
    ytt = Y[tr:tr+tt]
    return (xtr, xtt, ytr, ytt)


def remove_next_columns(df, column):
    cols = [col for col in df.columns if "next" not in col.lower()]
    cols.append(column)
    df = df[cols]
    return (df, column)


def remove_cp_columns(df):
    cols = [col for col in df.columns if not col.lower().startswith("cp")]
    df = df[cols]
    return df


def remove_previous_columns(df, column):
    cols = [col for col in df.columns if not col.lower().startswith("previous")]
    cols.append(column)
    df = df[cols]
    return df


def remove_max_avg_min_columns(df):
    cols = [col for col in df.columns if not (col.lower().startswith(
        "max") or col.lower().startswith("avg") or col.lower().startswith("min"))]
    df = df[cols]
    return df


def run_linear(X_train, X_test, Y_train, Y_test, num, col, security_code):
    linear_pipeline = Pipeline([("feature_selection", SequentialFeatureSelector(LinearRegression(
    ), n_jobs=None, n_features_to_select=num)), ("linear_regression", LinearRegression())])
    linear_pipeline.fit(X_train, Y_train)
    # pickle.dump(linear_pipeline, open(os.path.join(
    #     modelpath, str(security_code) + "_" + col + ".sav", 'wb')))
    Y_pred = linear_pipeline.predict(X_test)
    result = error_metrics(Y_test, Y_pred)
    selected_features = X_train.columns[linear_pipeline["feature_selection"].get_support(
    )].tolist()
    result.update({"selected_features": selected_features})
    result.update({"numoffeatures": len(selected_features)})
    result.update({"predicted_column": col})
    result.update({"model": "linear"})
    result.update({"actual": Y_test.values.tolist()})
    result.update({"predicted": Y_pred.tolist()})
    return result


def run_models(df, col, security_code):
    ref = df.copy()
    days = int(re.findall(r"\d+", col)[0])
    start = df['Date'].iloc[0] + datetime.timedelta(days=days)
    end = df['Date'].iloc[-1] - datetime.timedelta(days=days)
    df = df[df.Date.between(start, end)]
    df = pre_process_data(df, 60)
    df[df.columns] = (df[df.columns].astype(str)).apply(
        pd.to_numeric, errors='coerce')
    df, column = remove_next_columns(df, col)
    X = df.drop(columns=[column])
    Y = df[column]
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y, 0.70)
    num = 0.33
    linres = run_linear(X_train, X_test, Y_train, Y_test,
                        num, column, security_code)
    linres.update({"close": ref.loc[X_test.index]
                  ['Close Price'].values.tolist()})
    linres.update({"date": ref.loc[X_test.index]['Date'].apply(
        lambda row: row.strftime('%Y-%m-%d')).values.tolist()})

    return linres


@ray.remote
def run_companies_lb(security_code, col):
    try:
        print(security_code)
        security_code = str(security_code)
        df = pd.read_csv(os.path.join(path, "gr"+security_code+".csv"))
        df = df.iloc[::-1].reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[necessary_columns]
        result = run_models(df, col, security_code)
        result.update({"company": security_code})
        return result
    except:
        return None


@ray.remote
def run_companies_ub(security_code, col):
    try:
        print(security_code)
        security_code = str(security_code)
        df = pd.read_csv(os.path.join(path, "gr"+security_code+".csv"))
        df = df.iloc[::-1].reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[necessary_columns]
        result = run_models(df, col, security_code)
        result.update({"company": security_code})
        return result
    except:
        return None


def intial_run():
    fullresult = []
    for security_code in sp500companies:
        try:
            lbresult = run_companies_lb.remote(
                security_code, columns_to_predict[9])
            ubresult = run_companies_ub.remote(
                security_code, columns_to_predict[10])
            result = ray.get([lbresult, ubresult])
            if result[0] != None and result[1] != None:
                fullresult.append(result)
        except:
            traceback.print_exc()
    resultdf = pd.DataFrame(fullresult)
    resultdf.to_csv(os.path.join(os.getcwd(), "Data", "next_720_days.csv"), index=None)


necessary_columns = ["Date", "Close Price", "Previous 360 days UB", "Min Inc % in 180 days", "Next 60 days LB", "Previous 720 days UB", "No. of Trades GR", "CP % LV 180 days", "Max Inc % in 180 days", "Next 1080 days LB", "CP % BA 180 days", "Next Day Low Price GR", "Max Dec % in 90 days", "Expenditure GR", "CP % HV 90 days", "Min Dec % in 365 days", "Max Dec % in 365 days", "CP % HV 7 days", "CP % BA 7 days", "Avg Inc % in 365 days", "Min Inc % in 90 days", "Avg Inc % in 180 days", "Total Turnover (Rs.) GR", "Low Price GR", "Previous 1080 days UB", "CP % HV 180 days", "Next 180 days UB", "No.of Shares GR", "Previous 60 days UB", "CP % BA 90 days", "Avg Inc % in 90 days", "Sequential Increase %", "WAP GR", "CP % BA 30 days", "Avg Dec % in 180 days", "Previous 720 days LB", "EPS GR", "Deliverable Quantity GR", "Next 360 days UB", "CP % HV 365 days", "Spread Close-Open GR", "Min Dec % in 180 days", "Next 30 days LB", "Sequential Increase", "Previous 360 days LB",
                     "Alpha GR", "CP % LV 365 days", "Dividend Value GR", "Sequential Decrease", "Next 360 days LB", "Avg Dec % in 365 days", "Net Profit GR", "CP % LV 7 days", "CP % HV 30 days", "% Deli. Qty to Traded Qty GR", "Min Inc % in 365 days", "Sequential Decrease %", "Beta GR", "Next 30 days UB", "High Price GR", "Spread High-Low GR", "Income GR", "Max Dec % in 180 days", "Previous 30 days UB", "Next 90 days UB", "Next 90 days LB", "Next 1080 days UB", "Open Price GR", "Next 720 days LB", "Max Inc % in 365 days", "Previous 90 days LB", "Previous 90 days UB", "Next 60 days UB", "Avg Dec % in 90 days", "Previous 30 days LB", "Previous 1080 days LB", "Next Day Open Price GR", "Next Day High Price GR", "CP % BA 365 days", "Max Inc % in 90 days", "Revenue GR", "CP % LV 30 days", "Min Dec % in 90 days", "Next 180 days LB", "Previous 180 days LB", "Close Price GR", "CP % LV 90 days", "Previous 60 days LB", "Previous 180 days UB", "Next 720 days UB", "Next Day Close Price GR"]
columns_to_predict = ['Next 30 days LB', 'Next 30 days UB', 'Next 60 days LB', 'Next 60 days UB', 'Next 90 days LB', 'Next 90 days UB', 'Next 180 days LB',
                      'Next 180 days UB', 'Next 360 days LB', 'Next 360 days UB', 'Next 720 days LB', 'Next 720 days UB', 'Next 1080 days LB', 'Next 1080 days UB']

path = os.path.join(os.getcwd(), "Data", "GRStock")
sp500 = pd.read_csv(os.path.join(os.getcwd(), "Data", "SP500companies.csv"))
sp500companies = sp500['Security Code'].values.tolist()
sp500companies.sort()
ray.init(ignore_reinit_error=True)
intial_run()
