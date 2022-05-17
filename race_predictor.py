import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import warnings
from datetime import date
warnings.filterwarnings('ignore')
import statsmodels.api as sm

#pulling in missing dates
dates = pd.read_csv("Calendar.csv", sep=",")
dates = dates[["dt"]]
dates["dt"] = pd.to_datetime(dates["dt"]).dt.date

#Filtering dataset to aug/28 to todays date
start_date = pd.to_datetime('08-28-2021').date()
end_date = date.today()
dates = dates[dates["dt"] >= start_date]
dates = dates[dates["dt"] <= end_date]
dates["Date"] = dates["dt"]
dates = dates[["Date"]]

#Pulling in running data
data = pd.read_csv("garmindata.csv", sep=",")
data.drop(['Activity Type', 'Favorite', 'Title','Moving Time','Max HR','Max Run Cadence',
       'Avg Pace', 'Best Pace','Avg Vertical Ratio', 'Avg Vertical Oscillation',
       'Training Stress Score®', 'Grit', 'Flow', 'Dive Time', 'Min Temp',
       'Surface Interval', 'Decompression', 'Best Lap Time', 'Number of Laps',
       'Max Temp',  'Elapsed Time', 'Min Elevation',
       'Max Elevation'], axis=1, inplace=True)

#Cleaning running data
times = pd.to_timedelta(data['Time'])
dist = data["Distance"]
data["Date"] = pd.to_datetime(data["Date"]).dt.date
data["Avg Speed"] = dist / (times / pd.Timedelta('1 hour'))
data = data.replace(to_replace="--",value=0)
data["Calories"] = data["Calories"].str.replace(',','').astype(float)
data["Power"] = (data["Avg Speed"]/data["Avg HR"])*1000
data["Power"] = data["Power"].astype(float)
data.sort_values(by="Date",ascending=True,inplace=True)

#Joining date and urnning data together
data = pd.merge(dates,data, how='left')
data["Power"].fillna(data["Power"].mean(),inplace=True)
forecast_data = data[["Date","Power"]]
forecast_data["Date"] = forecast_data["Date"].astype(str)
forecast_data["Date"] = pd.to_datetime(forecast_data["Date"])

#creating model data
x_train = forecast_data[forecast_data["Date"] < '2022-04-01']
x_test = forecast_data[forecast_data["Date"] >= '2022-04-01']

#setting date to index
x_train.set_index("Date", inplace=True)
x_test.set_index("Date", inplace=True)

#finding best order
#stepwise_fit = auto_arima(data["Power"],trace=True,supress_warnings =True)
#print(stepwise_fit.summary())

#setting index
index_days = pd.date_range(x_train.index[-1], freq='D',periods=48)
model_arima = sm.tsa.arima.ARIMA(x_train, order=(4,2,5))
model = model_arima.fit()
fcast1 = model.forecast(48)
fcast1 = pd.Series(fcast1, index=index_days).dropna()
fcast1 = fcast1.rename("Arima")

#making preictions
start=len(x_train)
end=len(x_train)+len(x_test)-1
pred=model.predict(start=start,end=end,type='levels')

#plotting preictions
fig, ax = plt.subplots(figsize=(15,5))
chart = sns.lineplot(x="Date",y="Power", data = x_train)
chart.set_title('Training')
fcast1.plot(ax=ax, color ='red',marker='o',legend=True)
x_test.plot(ax=ax, color ='blue',marker='o',legend=True)
plt.show()

#How much is the model off by power on average
print("The MSE of ARIMA is: ", mean_squared_error(x_test["Power"].values,fcast1.values, squared=False))

#train model on entire dataset
model2 = sm.tsa.arima.ARIMA(data["Power"],order=(4,2,5))
model2=model2.fit()
t_date = date.today()
l_date = date(2022, 9, 25)
delta = l_date - t_date
index_future_dates = pd.date_range(start=date.today(),end='2022-09-25')
pred2=model2.predict(start=len(data),end=len(data)+delta.days,typ='levels').rename('ARIMA Predictions')
pred2.index=index_future_dates
mph = (pred2[-1]/1000)*170
print(mph)
print(round(mph,2), "MPH or",str((60/mph).astype(int))+":"+str((((60/mph)-10)*60).astype(int))+"/Mile")