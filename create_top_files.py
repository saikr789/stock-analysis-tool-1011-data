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
        df = df.drop(columns=["predicted"])
        df["minimum"] = df.apply(lambda row : row["actual"] if np.isnan(row["minimum"]) else row["minimum"], axis=1)
        df["maximum"] = df.apply(lambda row : row["actual"] if np.isnan(row["maximum"]) else row["maximum"], axis=1)
        df["min"] = df["actual"] - df["minimum"]
        df["max"] = df["maximum"] - df["actual"]
        buy = df.sort_values(by=["max"],ascending=[True])
        sell = df.sort_values(by=["max"],ascending=[False])
        buy.to_csv(os.path.join(toppath,"buy_"+str(days)+".csv"),index=None)
        sell.to_csv(os.path.join(toppath,"sell_"+str(days)+".csv"),index=None)
    except:
        pass
    time.sleep(5)