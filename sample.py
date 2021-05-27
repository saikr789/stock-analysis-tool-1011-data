import pandas as pd 
import os

path = os.path.join(os.getcwd(),"Data","GRStock")
cols = ["Revenue", "Dividend Value", "Income","Expenditure", "Net Profit", "EPS"]
colsgr = [i + " GR" for i in cols]

for name in os.listdir(path):
  df = pd.read_csv(os.path.join(path,name))
  for a in colsgr:
    df[a] = df[a].fillna(0)
  df.to_csv(os.path.join(path,name),index=None)
