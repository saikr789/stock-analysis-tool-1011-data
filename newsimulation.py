import os
import pandas as pd
import datetime
import re
import traceback


def simulation(df, investment, days, i):
    invest = False
    shares = 0
    df['date'] = pd.to_datetime(df['date'])
    start = df.iloc[-1]['date'] - datetime.timedelta(days=i)
    end = start - datetime.timedelta(days=days)
    refdf = df[df['date'].between(end, start)]
    simulation_result = []
    for _, row in refdf.iterrows():
        if row["invest"]:
            if invest is False:
                if investment < row['close']:
                    break

                shares = int(investment / row['close'])
                invested = shares * row['close']
                investment = investment - invested
                invest = True
                res = {"investment": invested, "shares": shares,
                       "entry": True, "exit": False, "date": row["date"].strftime("%d-%m-%Y"), "close": row["close"], "predictedub": row["predicted ub"]}
                simulation_result.append(res)
        if row['exit']:
            if invest is True:
                investment = investment + shares * row['close']
                res = {"investment": investment, "shares": shares,
                       "entry": False, "exit": True, "date": row["date"].strftime("%d-%m-%Y"), "close": row["close"], "predictedub": row["predicted ub"]}
                simulation_result.append(res)
                invest = False

    else:
        if invest is True:
            investment = investment + shares * row['close']
            invest = False
            res = {"investment": investment, "shares": shares,
                   "entry": False, "exit": True, "date": row["date"].strftime("%d-%m-%Y"), "close": row["close"], "predictedub": row["predicted ub"]}
            simulation_result.append(res)
    returns = []
    for i in range(0, len(simulation_result), 2):
        a = simulation_result[i]['investment']
        b = simulation_result[i+1]['investment']
        r = ((b-a)/a)
        returns.append(r)
    try:
        average_return_percent = sum(returns)/len(returns)
        return {"average_return_percent": average_return_percent, "simulation_result": simulation_result}
    except:
        pass


def perform(code):
    df = pd.read_csv(os.path.join(simpath, str(code)+".csv"))
    df['date'] = pd.to_datetime(df['date'])
    resdf = pd.DataFrame()
    columns = ["entry", "exit", "investment", "actual_returns", "predicted_returns",
               "actual_returns_percent", "predicted_returns_percent", "predicted_actual_percent_diff"]
    for i in range(0, 180):
        try:
            result = simulation(df, 100000, 180, i)
            simulation_result = result["simulation_result"]
            start = simulation_result[0]
            end = simulation_result[-1]
            predrets = start['predictedub'] * start['close'] * start['shares']
            actrets = end['close'] * end['shares']
            invst = start['investment']
            apercent = (actrets-invst)/invst
            ppercent = (predrets-invst)/invst
            res = [start['date'], end['date'], start['investment'],
                   actrets, predrets, apercent, ppercent, ppercent-apercent]
            resdf = resdf.append([res], ignore_index=True)
        except:
            pass
    resdf.columns = columns
    predrets = resdf.iloc[0]["investment"] + resdf.iloc[0]["investment"] * \
        resdf["predicted_returns_percent"].mean()
    finres = simulation(df, 100000, 180, 0)
    finres.update({"actual_returns": resdf["actual_returns"].iloc[0]})
    finres.update({"predicted_returns": predrets})
    finres.update(
        {"predicted_returns_percent": resdf["predicted_returns_percent"].mean()})
    return finres


simpath = os.path.join(os.getcwd(), "Data", "Simulation")

sp500 = pd.read_csv(os.path.join(os.getcwd(), "Data", "SP500companies.csv")).set_index("Security Code")

myres = []
for code in os.listdir(simpath):
    try:
        res = perform(int(code[:-4]))
        res.update({"code": code})
        company = re.sub('[!@#$%^&*(.)-=,\\\/\']','', sp500.loc[int(code), "Security Name"]).upper()
        res.update({"company": company})
        myres.append(res)
    except:
        pass
myresdf = pd.DataFrame(myres)
myresdf.to_csv(os.path.join(simpath, "myresdf.csv"), index=None)
