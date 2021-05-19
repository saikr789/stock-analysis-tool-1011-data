import pandas as pd
import os

path = os.path.join(os.getcwd(), "Data")
resultdf = pd.DataFrame()
if os.path.exists(os.path.join(path, "predict100.csv")):
    predict100 = pd.read_csv(os.path.join(path, "predict100.csv"))
    resultdf = resultdf.append(predict100)
if os.path.exists(os.path.join(path, "predict200.csv")):
    predict200 = pd.read_csv(os.path.join(path, "predict200.csv"))
    resultdf = resultdf.append(predict200)
if os.path.exists(os.path.join(path, "predict300.csv")):
    predict300 = pd.read_csv(os.path.join(path, "predict300.csv"))
    resultdf = resultdf.append(predict300)
if os.path.exists(os.path.join(path, "predict400.csv")):
    predict400 = pd.read_csv(os.path.join(path, "predict400.csv"))
    resultdf = resultdf.append(predict400)
if os.path.exists(os.path.join(path, "predict500.csv")):
    predict500 = pd.read_csv(os.path.join(path, "predict500.csv"))
    resultdf = resultdf.append(predict500)

resultdf = resultdf.dropna(how="all")
resultdf.to_csv(os.path.join(os.getcwd(), "Data",
                "next_30_days.csv"), index=None)

# path = os.path.join(os.getcwd(),"Data")
# predict100 = pd.read_csv(os.path.join(path,"predict100.csv"))
# predict200 = pd.read_csv(os.path.join(path,"predict200.csv"))
# predict300 = pd.read_csv(os.path.join(path,"predict300.csv"))
# predict400 = pd.read_csv(os.path.join(path,"predict400.csv"))
# predict500 = pd.read_csv(os.path.join(path,"predict500.csv"))

# resultdf = resultdf.append(predict100)
# resultdf = resultdf.append(predict200)
# resultdf = resultdf.append(predict300)
# resultdf = resultdf.append(predict400)
# resultdf = resultdf.append(predict500)
# resultdf = resultdf.set_index(drop=True)
# resultdf = resultdf.dropna(how="all")
# resultdf.to_csv(os.path.join(os.getcwd(), "Data","next_30_days.csv"), index=None)
