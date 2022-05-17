import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
import seaborn as sns

#pulling in data
data = pd.read_csv("garmindata.csv", sep=",")
data.drop(['Activity Type', 'Date', 'Favorite', 'Title','Time','Max HR','Max Run Cadence',
       'Avg Pace', 'Best Pace','Avg Vertical Ratio', 'Avg Vertical Oscillation',
       'Training Stress Score®', 'Grit', 'Flow', 'Dive Time', 'Min Temp',
       'Surface Interval', 'Decompression', 'Best Lap Time', 'Number of Laps',
       'Max Temp',  'Elapsed Time', 'Min Elevation',
       'Max Elevation'], axis=1, inplace=True)
times = pd.to_timedelta(data['Moving Time'])
dist = data["Distance"]
data["Avg Speed"] = dist / (times / pd.Timedelta('1 hour'))
data = data.replace(to_replace="--",value=0)
data["Calories"] = data["Calories"].str.replace(',','').astype(float)
data = data[["Distance", "Calories", "Avg HR", "Avg Run Cadence", "Avg Speed", "Avg Stride Length","Total Ascent", "Total Descent"]]
data =data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
predict = "Avg Speed"

X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.3)


"""best = 0
for _ in range(50):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.3)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    if acc > best:
        best == acc
        with open("run_model.pickle", "wb") as f:
            pickle.dump(linear, f)"""

pickle_in = open("run_model.pickle", "rb")
linear = pickle.load(pickle_in)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = "Avg HR"
style.use("ggplot")
pyplot.scatter(data[p], data["Avg Speed"])
pyplot.xlabel(p)
pyplot.ylabel("Avg Speed")
pyplot.show()

sns.heatmap(data.corr(), cmap="YlGnBu", annot = True)
pyplot.show()

race_predictor = pd.DataFrame([[26.2, 2620, 170, 165, .91, 278, 275]])
race_predictions = linear.predict(race_predictor)
print("Expected Race Pace: ", str(race_predictions)[1:-1], "MPH or ",str((60/race_predictions).astype(int))[1:-1]+":"+str((((60/race_predictions)-10)*60).astype(int))[1:-1]+"/mile")