import os
import pandas as pd
import re
import traceback
import time
pd.options.mode.chained_assignment = None

def create_files(filename, days):
    df = pd.read_csv(os.path.join(os.getcwd(), "Data", filename))
    df = df.dropna(how="all")
    simpath = os.path.join(os.getcwd(), "Data", "Simulation")
    if not os.path.exists(simpath):
        os.makedirs(simpath)

    sp500 = pd.read_csv(os.path.join(
        os.getcwd(), "Data", "SP500companies.csv")).set_index("Security Code")
    cols = ['predicted_column', 'actual',
            'predicted', 'close', 'date', 'company']
    df = df[cols]

    df['company'] = df['company'].apply(
        lambda row: str(int(row)) + "-" + re.sub('[!@#$%^&*(.)-=,\\\/\']', '', sp500.loc[int(row), "Security Name"]).upper())
    for n, g in df.groupby(by=['company']):
        lower = g.iloc[0]
        upper = g.iloc[1]

        date = [d.strip()
                for d in lower['date'][1:-1].replace("\'", "").split(",")]
        close = [float(c.strip()) for c in lower['close'][1:-1].split(",")]
        actual_lb = [float(a.strip())
                     for a in lower['actual'][1:-1].split(",")]
        predicted_lb = [float(p.strip())
                        for p in lower['predicted'][1:-1].split(",")]
        actual_ub = [float(a.strip())
                     for a in upper['actual'][1:-1].split(",")]
        predicted_ub = [float(p.strip())
                        for p in upper['predicted'][1:-1].split(",")]

        cols = ["date", "close", "actual lb",
                "predicted lb", "actual ub", "predicted ub"]
        refdf = pd.DataFrame(zip(date, close, actual_lb,
                                 predicted_lb, actual_ub, predicted_ub), columns=cols)

        refdf["actual ub close diff"] = abs(
            refdf["close"] - refdf["close"] * refdf["actual ub"])
        refdf["predicted ub close diff"] = abs(
            refdf["close"] - refdf["close"] * refdf["predicted ub"])
        refdf["actual lb close diff"] = abs(
            refdf["close"] - refdf["close"] * refdf["actual lb"])
        refdf["predicted lb close diff"] = abs(
            refdf["close"] - refdf["close"] * refdf["predicted lb"])
        refdf["predicted lb ub diff"] = refdf["predicted ub close diff"] - \
            refdf["predicted lb close diff"]
        refdf["predicted lb %"] = 1 - refdf["predicted lb"]
        refdf["predicted ub %"] = refdf["predicted ub"] - 1
        refdf["invest"] = refdf.apply(lambda row: True if row["predicted lb %"] < 0.01 and (
            row["predicted ub %"] - row["predicted lb %"]) > 0.1 else False, axis=1)
        refdf["exit"] = refdf.apply(lambda row: True if row["predicted ub %"] < 0.01 and (
            row["predicted ub %"] + row["predicted lb %"]) > 0.05 else False, axis=1)
        refdf.to_csv(os.path.join(simpath, str(
            n[:6])+"_"+str(days)+".csv"), index=None)


result = []
for days in [30, 60, 90, 180, 270, 360, 540, 720, 900, 1080]:
    try:
        filename = "next" + "_" + str(days) + "_" + "days" + ".csv"
        create_files(filename, days)
    except:
        traceback.print_exc()
time.sleep(100) 
