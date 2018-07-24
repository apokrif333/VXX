import pandas as pd
import numpy as np
import matplotlib as mp
import datetime
from datetime import datetime
import more_itertools as mit
import statistics as stat
import math

StartCapital = 10000000
Commissions = 0.0050

def DateCheck(x):
    try:
        x["Date"] = pd.to_datetime(x["Date"], format="%Y-%m-%d")
    except:
        x["Date"] = pd.to_datetime(x["Date"], format="%m/%d/%Y")

# start = "2010-02-11"
# end = "2018-06-04" #datetime.today().date()
# TQQQBase = web.DataReader("TQQQ", "morningstar", start, end)
# TQQQBase.to_csv("TQQQ"+".csv")

VIXBase = pd.read_csv("^VIX.csv")
VXVBase = pd.read_csv("^VXV.csv")
VXXBase = pd.read_csv("VXX.csv")
DateCheck(VIXBase)
DateCheck(VXVBase)
DateCheck(VXXBase)

ORatio = []
for i in range(0, len(VIXBase["Date"])):
    ORatio.append(VIXBase["Open"].iloc[i]/VXVBase["Open"].iloc[i])

ATR =[]
for i in range(0,len(VXXBase["Date"])):
    if i>=10 and len(ATR)>=10:
        curVXX = VXXBase.head(i)
        curVXX = curVXX.tail(10)
        ATR.append(np.average(curVXX["High"]-curVXX["Low"].tolist()))
    else:
        ATR.append(0)

#Глобальные массивы, которые сохранятся после всех расчётов
Ratio = []
Stop = []
CAGR = []
StDev = []
DrawDown = []
Sharpe = []
MaR = []
SM = []

for MaxRatio in mit.numeric_range(0.88, 1.21, 0.001):
    for MaxStop in mit.numeric_range(0.0, 5, 0.01):
        #Массивы текущего блока расчётов
        Enter = []
        EnterPrice = []
        Capital = []
        Shares = []
        DayCng = []
        Down = []

        MaxRatio = round(MaxRatio,3)
        MaxStop = round(MaxStop,2)

        for i in range(0, len(ORatio)):
            if MaxRatio < ORatio[i] or ATR[i] == 0:
                Enter.append(0)
            elif MaxRatio > ORatio[i] and VXXBase["Open"][i]+MaxStop*ATR[i] <= VXXBase["High"][i]:
                Enter.append(-1)
            else:
                Enter.append(1)
        #Блок расчёта капитала
        for i in range (0, len(Enter)):
            #Если самая первая строчка
            if i == 0:
                Shares.append(0)
                Capital.append(StartCapital)
                EnterPrice.append(0)
            #Если вчера ещё были в позиции, а сегодня надо выходить
            elif Enter[i] == 0 and Enter[i-1] == 1:
                Shares.append(0)
                Capital.append(Shares[i-1]*EnterPrice[-1]+(EnterPrice[-1]-VXXBase["Open"][i])*Shares[i-1]
                               -Shares[i-1]*Commissions)
                EnterPrice.append(0)
            #Если вчера и сегодня нужно быть вне позиции
            elif Enter[i] == 0 and (Enter[i-1] == 0 or Enter[i-1] == -1):
                Shares.append(0)
                Capital.append(Capital[i-1])
                EnterPrice.append(0)
            #Если получили стоп, но зашли в позу только сегодня или вчера получили стоп
            elif Enter[i] == -1 and (Enter[i-1] == 0 or Enter[i-1] == -1):
                Shares.append(Capital[i-1]/VXXBase["Open"][i])
                EnterPrice.append(VXXBase["Open"][i])
                Capital.append(Shares[i]*EnterPrice[-1]+(EnterPrice[-1]-(VXXBase["Open"][i]+MaxStop*ATR[i]))*Shares[i]
                               -Shares[i]*Commissions*2)
            #Если получили стоп, но были в позе ранее
            elif Enter[i] == -1 and Enter[i-1] == 1:
                Shares.append(Shares[i-1])
                Capital.append(Shares[i]*EnterPrice[-1]+(EnterPrice[-1]-(VXXBase["Open"][i]+MaxStop*ATR[i]))*Shares[i]
                               -Shares[i]*Commissions)
                EnterPrice.append(0)
            #Если вчера были вне позици, а сегодня надо входить
            elif Enter[i] == 1 and (Enter[i-1] == 0 or Enter[i-1] == -1):
                Shares.append(Capital[i-1]/VXXBase["Open"][i])
                Capital.append(Capital[i-1]-Shares[i]*Commissions)
                EnterPrice.append(VXXBase["Open"][i])
            #Если сегодня спокойно сидим в позиции
            elif Enter[i] == 1:
                Shares.append(Shares[i-1])
                Capital.append(Shares[i]*EnterPrice[-1]+(EnterPrice[-1]-VXXBase["Open"][i])*Shares[i])
                EnterPrice.append(EnterPrice[-1])

        for i in range(0, len(Capital)):
            if i == 0:
                DayCng.append(0)
            else:
                DayCng.append(Capital[i]/Capital[i-1]-1)

        High = 0
        for i in range(0, len(Capital)):
            if Capital[i]>High:
                High = Capital[i]
            Down.append((Capital[i]/High-1)*100)

        Ratio.append(MaxRatio)
        Stop.append(MaxStop)
        CAGR.append(((Capital[-1]/Capital[0])**
                 (1/(VXXBase["Date"].iloc[-1].year-VXXBase["Date"].iloc[0].year))-1)*100)
        StDev.append(stat.stdev(DayCng)*math.sqrt(252))
        DrawDown.append(min(Down))
        Sharpe.append(CAGR[-1]/StDev[-1])
        MaR.append(abs(CAGR[-1]/DrawDown[-1]))
        SM.append(Sharpe[-1]*MaR[-1])

        print(MaxRatio)
        print(MaxStop)

exportTable = pd.DataFrame({"Ratio": Ratio,
                            "Stop": Stop,
                            "CAGR": CAGR,
                            "StDev": StDev,
                            "DrawDown": DrawDown,
                            "Sharpe": Sharpe,
                            "MaR": MaR,
                            "SM": SM},
                           columns=["Ratio", "Stop", "CAGR", "StDev", "DrawDown", "Sharpe", "MaR", "SM"]
                           )

exportTable.to_csv("C:/Users/Tom/Google Диск/Прог/VXX_Metric.csv")