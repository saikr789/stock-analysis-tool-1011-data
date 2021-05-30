import os
import pandas as pd
import numpy as np
import time

toppath = os.path.join(os.getcwd(),"Data","Top")

for days in [30,60,90,180,360,720]:
    try:
        print(days)
        spath = "simres_{}.csv".format(days)
        df = pd.read_csv(os.path.join(toppath,spath))
        df["minimum"] = df.apply(lambda row : row["actual"] if np.isnan(row["minimum"]) else row["minimum"], axis=1)
        df["maximum"] = df.apply(lambda row : row["actual"] if np.isnan(row["maximum"]) else row["maximum"], axis=1)
        df[["minimum","maximum"]] = df.apply(lambda row : pd.Series([np.nanmin([row["minimum"],row["maximum"]]),np.nanmax([row["minimum"],row["maximum"]])]),axis=1)
        df["min"] = df["actual"] - df["minimum"]
        df["max"] = df["maximum"] - df["actual"]
        df["suggest"] = df.apply(lambda row: "buy" if row["minimum"] <= row["actual"] <= row["minimum"] * 1.15 else "sell" if row["maximum"] * 0.85 <= row["actual"] <= row["maximum"] else None, axis=1)
        sell = df.sort_values(by=["max"], ascending=[True])
        buy = df.sort_values(by=["min"], ascending=[True])
        df.to_csv(os.path.join(toppath,spath),index=None)
        buy.to_csv(os.path.join(toppath,"buy_"+str(days)+".csv"),index=None)
        sell.to_csv(os.path.join(toppath,"sell_"+str(days)+".csv"),index=None)
    except Exception as e:
        print(e)
    time.sleep(5)
