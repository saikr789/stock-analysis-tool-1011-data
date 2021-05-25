import os
import pandas as pd
import datetime
import re
import traceback
pd.options.mode.chained_assignment = None

def simulation(df, investment, days):
    invest = False
    shares = 0
    df['date'] = pd.to_datetime(df['date'])
    start = df.iloc[-1]['date']
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
                       "entry": True, "exit": False, "date": row["date"].strftime("%d-%m-%Y"), "close": row["close"]}
                simulation_result.append(res)
        if row['exit']:
            if invest:
                investment = investment + shares * row['close']
                res = {"investment": investment, "shares": shares,
                       "entry": False, "exit": True, "date": row["date"].strftime("%d-%m-%Y"), "close": row["close"]}
                simulation_result.append(res)
                invest = False

    else:
        if invest:
            investment = investment + shares * row['close']
            invest = False
            res = {"investment": investment, "shares": shares,
                   "entry": False, "exit": True, "date": row["date"].strftime("%d-%m-%Y"), "close": row["close"]}
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
        return None

def simulate(investment, days):
    topreturns = []
    for security_code in sp500.index.tolist():
        try:
            security_code = str(security_code)
            spath = security_code + "_" + str(days) + ".csv"
            df = pd.read_csv(os.path.join(simpath, spath))
            company = re.sub('[!@#$%^&*(.)-=,\\\/\']', '',
                             sp500.loc[int(security_code), "Security Name"]).upper()
            result = simulation(df, investment, days)
            if result == None:
                continue
            result.update({"company": company})
            result.update({"code": security_code})
            topreturns.append(result)
            simdf = pd.DataFrame(result["simulation_result"])
            simdf.to_csv(os.path.join(simrespath, spath), index=None)
            print(security_code+"-"+company)
        except:
            pass
    if topreturns == []:
        return
    cols = ["company","code","average_return_percent","simulation_result"]
    topreturnscompanies = pd.DataFrame(topreturns)
    topreturnscompanies = topreturnscompanies[cols]
    topreturnscompanies = topreturnscompanies.sort_values(by=["average_return_percent"], ascending=[False])
    topreturnscompanies.to_csv(os.path.join(simrespath, "top_" + str(days)+".csv"), index=None)


sp500 = pd.read_csv(os.path.join(os.getcwd(), "Data",
                    "SP500companies.csv")).set_index("Security Code")

simpath = os.path.join(os.getcwd(), "Data", "Simulation")
simrespath = os.path.join(os.getcwd(), "Data", "SimulationResult")

if not os.path.exists(simrespath):
    os.makedirs(simrespath)

investment = 100000
for days in [30, 60, 90, 180, 360, 720, 900, 1080]:
    try:
        print(days)
        simulate(investment, days)
    except:
        traceback.print_exc()

time.sleep(100)